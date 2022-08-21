#![feature(future_join, future_poll_fn)]

use async_mutex::Mutex;
use futures::executor::block_on;
use std::fmt::Debug;
use std::future::{Future, join};
use std::mem::{self, MaybeUninit};
use std::ptr;
use std::sync::Arc;
use time::ext::NumericalDuration;
use time::{Duration, OffsetDateTime};
use tracing::{instrument, debug};

/// In-memory temporary value cache with automatic refresh.
/// 
/// # Safety
/// This cache currently has a memory safety hole where [`Cache::read`] is
/// capable of swapping out the underlying value while references to that value
/// still exist. Use caution when using `Cache` in a multi-threaded context.
#[derive(Debug)]
pub struct Cache<Item, ItemSource>
where
	Item: Debug,
{
	value: Mutex<CacheBox<Item>>,
	source: Arc<Mutex<ItemSource>>,
}

impl<Item, Err, Fut, ItemSource> Cache<Item, ItemSource>
where
	Item: Debug,
	Err: Debug,
	Fut: Future<Output = Result<Item, Err>>,
	ItemSource: FnMut() -> Fut,
{
	/// Construct an empty cache whose value expires after `lifetime` minutes
	/// past its creation.
	pub fn new(lifetime: i64, source: ItemSource) -> Self {
		assert!(lifetime > 0, "cache cannot expire in the past!");
		
		Cache {
			value: Mutex::new(CacheBox::new(lifetime)),
			source: Arc::new(Mutex::new(source)),
		}
	}

	/// Construct a pre-initialised cache with a default value, which expires
	/// after `lifetime` minutes past its creation.
	pub fn with_value(lifetime: i64, value: Item, source: ItemSource) -> Self {
		assert!(lifetime > 0, "cache cannot expire in the past!");

		Cache {
			value: Mutex::new(CacheBox::with_value(lifetime, value)),
			source: Arc::new(Mutex::new(source))
		}
	}

	/// Immediately invalidate the value inside the cache slot, causing it to
	/// be re-acquired the next time it is read.
	pub async fn invalidate_now(&self) {
		let mut value = self.value.lock().await;
		value.invalidate_now();
	}

	/// Read the current value in the cache slot, automatically invalidating and
	/// re-acquiring it if necessary.
	#[instrument(skip(self))]
	pub async fn read<'value>(&'value self) -> Result<&'value Item, Err> {
		use async_mutex::MutexGuard;

		let value = self.value.lock();
		let source = self.source.lock();

		// wait concurrently until both locks are acquired, instead of waiting
		// for them one at a time.
		// UNSTABLE: feature(future_join, future_poll_fn)
		let (mut value, mut source): (MutexGuard<CacheBox<Item>>, MutexGuard<ItemSource>) = join!(value, source).await;

		if !value.is_valid() {
			debug!(expires_at = %value.expires_at, "cache is invalid; updating");
			value.acquire_value(&mut *source).await?;
		};

		// SAFETY: it is not possible to construct a CacheBox<T> in a state
		// that `now < expires_at` before the cache is initialised.
		// 
		// If initialisation fails, `acquire_value` exits out before it updates
		// `expires_at`. Also, this function exits out before it reaches this
		// point.
		Ok(unsafe { &*value.as_ptr() })
	}

	/// Read the current value in the cache slot mutably, automatically
	/// invalidating and re-acquiring it if necessary.
	#[instrument(skip(self))]
	pub async fn read_mut<'value>(&'value mut self) -> Result<&'value mut Item, Err> {
		use async_mutex::MutexGuard;

		let value = self.value.lock();
		let source = self.source.lock();

		// wait concurrently until both locks are acquired, instead of waiting
		// for them one at a time.
		// UNSTABLE: feature(future_join, future_poll_fn)
		let (mut value, mut source): (MutexGuard<CacheBox<Item>>, MutexGuard<ItemSource>) = join!(value, source).await;

		if !value.is_valid() {
			debug!(expires_at = %value.expires_at, "cache is invalid; updating");
			value.acquire_value(&mut *source).await?;
		};

		// SAFETY: it is not possible to construct a CacheBox<T> in a state
		// that `now < expires_at` before the cache is initialised.
		// 
		// If initialisation fails, `acquire_value` exits out before it updates
		// `expires_at`. Also, this function exits out before it reaches this
		// point.
		// 
		// Acquiring a mutable pointer from `self.value` requires being able to
		// borrow `self` mutably, which will fail if another mutable reference
		// exists.
		Ok(unsafe { &mut *value.as_mut_ptr() })
	}
}

impl<Item, ItemSource> Clone for Cache<Item, ItemSource>
where
	Item: Debug + Clone,
{
	fn clone(&self) -> Self {
		let value = Mutex::new(block_on(self.value.lock()).clone());
		let source = self.source.clone();

		Cache {
			value,
			source,
		}
	}
}

/// Storage utility for cache items. Used as an implementation detail of
/// [`Cache<Item, ItemStorage>`].
#[derive(Debug)]
struct CacheBox<Item>
where
	Item: Debug,
{
	item: MaybeUninit<Item>,

	// flag used to indicate whether `item` contains an initialised value, for
	// drop safety checking.
	is_initialized: bool,

	expires_at: OffsetDateTime,
	lifetime: Duration,
}

impl<Item> CacheBox<Item>
where
	Item: Debug,
{
	/// Helper function to get the current date and time.
	fn current_datetime() -> OffsetDateTime {
		OffsetDateTime::now_utc()
	}

	/// Construct an empty cache slot whose value expires after `lifetime`
	/// minutes past its creation.
	fn new(lifetime: i64) -> Self {
		CacheBox {
			item: MaybeUninit::uninit(),
			
			expires_at: OffsetDateTime::UNIX_EPOCH,
			lifetime: lifetime.minutes(),

			is_initialized: false,
		}
	}

	/// Construct a pre-initialised cache slot with a default value, which
	/// expires after `lifetime` minutes past its creation.
	fn with_value(lifetime: i64, value: Item) -> Self {
		CacheBox {
			item: MaybeUninit::new(value),
			
			expires_at: OffsetDateTime::now_utc() + lifetime.minutes(),
			lifetime: lifetime.minutes(),

			is_initialized: true,
		}
	}

	/// Gets a pointer to the cached item.
	/// 
	/// # Safety
	/// 
	/// Reading from this pointer or turning it into a reference is undefined
	/// behaviour unless the cache contains a valid `Item` instance.
	/// 
	/// Caller should ensure that [`CacheBox::acquire_value`] has been called
	/// at least once and returned a successful result, or the `CacheBox` was
	/// constructed with [`CacheBox::from_value`], prior to calling this
	/// function.
	/// 
	/// Caller should ensure this pointer is never used to create a mutable
	/// reference.
	fn as_ptr(&self) -> *const Item {
		self.item.as_ptr()
	}

	/// Gets a mutable pointer to the cached item.
	/// 
	/// # Safety
	/// 
	/// Reading from this pointer or turning it into a reference is undefined
	/// behaviour unless the cache contains a valid `Item` instance.
	/// 
	/// Caller should ensure that [`CacheBox::acquire_value`] has been called
	/// at least once and returned a successful result, or the `CacheBox` was
	/// constructed with [`CacheBox::from_value`], prior to calling this
	/// function.
	/// 
	/// Caller should ensure that no other mutable references to the cached item
	/// exist before calling this function.
	fn as_mut_ptr(&mut self) -> *mut Item {
		self.item.as_mut_ptr()
	}

	/// Acquire a new value from the provided item source and store it in the
	/// cache slot, overwriting the old value if necessary.
	#[instrument(
		name = "Acquiring an updated value from the item source",
		skip(self, source),
		fields(
			expired_at = %self.expires_at,
			lifetime = %self.lifetime,
		),
	)]
	async fn acquire_value<Err, Fut, ItemSource>(&mut self, source: &mut ItemSource) -> Result<(), Err>
	where
		Err: Debug,
		Fut: Future<Output = Result<Item, Err>>,
		ItemSource: FnMut() -> Fut,
	{
		let value = source().await?;

		self.drop_if_needed();
		let ptr = self.item.as_mut_ptr();

		// SAFETY: we know this pointer is valid, as we just acquired it from
		// our slot.
		unsafe {
			ptr::write(ptr, value);
		}

		self.expires_at = Self::current_datetime() + self.lifetime;
		self.is_initialized = true;

		debug!("cache expiry time updated: {expires_at}", expires_at = self.expires_at);
		Ok(())
	}

	/// Immediately invalidate the value inside the cache slot.
	#[instrument(
		name = "Invalidating cache now",
		skip(self),
	)]
	fn invalidate_now(&mut self) {
		// cache is considered invalid if `now >= expires_at`
		self.expires_at = Self::current_datetime();
	}

	/// Check if the cache is considered valid.
	/// 
	/// A cache is considered valid if `now < expires_at`, and is invalidated
	/// automatically when time progresses such that this is no longer true.
	fn is_valid(&self) -> bool {
		// cache is considered invalid if `now >= expires_at`
		Self::current_datetime() < self.expires_at
	}

	/// Drop the inner value if it is safe to.
	/// 
	/// This checks if `Item` should be dropped and, if so, whether we contain
	/// an initialised value. If these are both true, it drops the contained
	/// value.
	fn drop_if_needed(&mut self) {
		if mem::needs_drop::<Item>() && self.is_initialized {
			// SAFETY: it is not possible to construct a CacheBox<T> in a
			// state that `is_initialized == true` when `self.item` is
			// uninitialised.
			unsafe {
				ptr::drop_in_place(self.item.as_mut_ptr());
			}
		}

		self.is_initialized = false;
	}
}

impl<Item> Clone for CacheBox<Item>
where
	Item: Debug + Clone,
{
	fn clone(&self) -> Self {
		let expires_at = self.expires_at.clone();
		let lifetime = self.lifetime.clone();
		let is_initialized = self.is_initialized;

		let item = {
			let mut container = MaybeUninit::uninit();

			// use self.is_valid() instead of self.is_initialized as there is
			// no point cloning a stale value.
			if self.is_valid() {
				// SAFETY: it is not possible to construct a CacheBox<T> in a
				// state that `now < expires_at` when self.item is
				// uninitialised.
				unsafe {
					let value = (&*self.item.as_ptr()).clone();
		
					ptr::write(container.as_mut_ptr(), value);
				}
			}

			container
		};

		CacheBox {
			item,
			is_initialized,

			expires_at,
			lifetime,
		}
	}
}

impl<Item> Drop for CacheBox<Item>
where
	Item: Debug,
{
	fn drop(&mut self) {
		// if we are initialised, drop the `T` in `self.item`
		self.drop_if_needed();
	}
}

fn _cache_box_must_be_sync<Item>()
where
	Item: Debug,
	CacheBox<Item>: Send + Sync {}

fn _cache_must_be_sync<Item, ItemSource>()
where
	Item: Debug,
	ItemSource: FnMut() -> Item,
	Cache<Item, ItemSource>: Send + Sync {}

fn main() {
	#[allow(dead_code)]
	#[derive(Debug, Clone)]
	struct Character {
		id: u64,
		name: String,
	}

	let cache = Cache::new(30, || async {
		Ok::<_, ()>(Character { id: 0, name: String::from("togglebit") })
	});

	let value = block_on(cache.read());
	println!("{value:?}");
}

#[cfg(test)]
mod test {
	use super::Cache;
	use async_mutex::Mutex;
	use futures_test::test;
	use std::sync::atomic::{AtomicU8, Ordering};

	#[test]
	async fn cache_updates_successfully() {
		let counter = Mutex::new(0);
		let cache = Cache::new(1, || async {
			let mut counter = counter.lock().await;
			*counter += 1;

			Ok::<_, ()>(format!("retrieved {counter} times"))
		});

		let actual = cache.read().await.cloned();
		assert_eq!(actual.as_deref(), Ok("retrieved 1 times"));

		cache.invalidate_now().await;

		let actual = cache.read().await.cloned();
		assert_eq!(actual.as_deref(), Ok("retrieved 2 times"));
	}

	#[test]
	async fn cache_basic_drop_code_executes_successfully() {
		static DROPS: AtomicU8 = AtomicU8::new(0);

		#[derive(Debug)]
		struct ShouldDrop;
		impl Drop for ShouldDrop {
			fn drop(&mut self) {
				DROPS.fetch_add(1, Ordering::SeqCst);
			}
		}

		let cache = Cache::new(1, || async {
			Ok::<ShouldDrop, ()>(ShouldDrop)
		});

		let _ = cache.read().await;
		cache.invalidate_now().await;
		let _ = cache.read().await;
		cache.invalidate_now().await;
		let _ = cache.read().await;

		assert_eq!(DROPS.load(Ordering::SeqCst), 2);
	}
}
