#ifndef TIMED_ATOMIC_HPP
#define TIMED_ATOMIC_HPP

/*
Author: Loukas Kordos
Disclaimer: This code was proudly written without LLMs :)
*/

/*
Some caveats:
- I know that this will not work for extremely high frequency due to OS scheduler frequency often being 1ms, so anything below that is unreliable. But
for my use cases here (<500Hz), I ran this for a long time without any issues and checked th waiting time using the tracy profiler, it never had any
issues. If I was to rewrite this, I would just use a wait-free data structure like triple buffering or seqlock to fix this.
- Even though the try_lock_for will voluntarily yield and thus potentially exceed the deadline slightly, I pinned the thread to a CPU, isolated it
from the kernel and any other processes, and increased its priority. This mitigated the effect that yielding can have because no scheduling timer is
running on that thread so the only thing that it has to wait for is the kernel hrtimer.
- Regarding more complex objects and unbounded execution time due to heap operations: I am aware, and this is why I used atomic shared pointers for
larger data structures, to avoid exactly this. I am aware that dropping a shared_ptr can trigger a heap deallocation (`delete`) on the consumer thread
if it holds the last reference, but this data operates at a much lower frequency where the occasional heap overhead is acceptable.. So I only used
timed_atomic for small to medium data on the stack.
*/

#include <chrono>
#include <expected>
#include <memory>
#include <mutex>
#include <type_traits>
#include <variant>

template <typename T>
struct is_shared_ptr : std::false_type
{
};

template <typename U>
struct is_shared_ptr<std::shared_ptr<U>> : std::true_type
{
};

template <typename T>
struct is_unique_ptr : std::false_type
{
};

template <typename U, typename D>
struct is_unique_ptr<std::unique_ptr<U, D>> : std::true_type
{
};

template <typename T>
struct is_raw_pointer : std::is_pointer<T>
{
};

template <typename T>
concept NotPointerType = !is_raw_pointer<T>::value && !is_unique_ptr<T>::value && !is_shared_ptr<T>::value;

template <typename T>
concept AllowedType = NotPointerType<T> && std::is_copy_constructible_v<T> && std::is_move_constructible_v<T>;

/// @brief Wrapper class to allow threadsafe access with a timeout.
template <typename T>
    requires AllowedType<T>
class timed_atomic
{
private:
    mutable std::timed_mutex mtx{};
    T value{};

public:
    explicit timed_atomic(T initial_value) : value{initial_value} {}
    timed_atomic() = default;

    std::expected<T, std::monostate> try_load_for(std::chrono::microseconds timeout_us) const
    {
        std::unique_lock lock{mtx, std::defer_lock};
        if (!lock.try_lock_for(timeout_us)) { return std::unexpected{std::monostate{}}; }
        return value;
    }

    bool try_store_for(T new_value, std::chrono::microseconds timeout_us)
    {
        std::unique_lock lock{mtx, std::defer_lock};
        if (!lock.try_lock_for(timeout_us)) { return false; }
        value = std::move(new_value);
        return true;
    }
};

#endif