#ifndef TIMED_ATOMIC_HPP
#define TIMED_ATOMIC_HPP

#include <mutex>
#include <chrono>
#include <expected>
#include <memory>
#include <type_traits>

template<typename T>
struct is_shared_ptr : std::false_type {};

template <typename U>
struct is_shared_ptr<std::shared_ptr<U>> : std::true_type {};

template <typename T>
struct is_unique_ptr : std::false_type {};

template <typename U, typename D>
struct is_unique_ptr<std::unique_ptr<U, D>> : std::true_type {};

template <typename T>
struct is_raw_pointer : std::is_pointer<T> {};

template <typename T>
concept NotPointerType = !is_raw_pointer<T>::value && !is_unique_ptr<T>::value && !is_shared_ptr<T>::value;

template<typename T>
concept AllowedType = NotPointerType<T> && std::is_copy_constructible_v<T> && std::is_move_constructible_v<T>;

/// @brief Wrapper class to allow threadsafe access with a timeout.
template<typename T>
requires AllowedType<T>
class timed_atomic {
    private:
        std::timed_mutex mtx {};
        T value {};

    public:
        explicit timed_atomic(T initial_value) : value{initial_value} {}
        timed_atomic() = default;

        std::expected<T, std::monostate> try_load_for(std::chrono::microseconds timeout_us) {
            std::unique_lock lock{mtx, std::defer_lock};
            if(!lock.try_lock_for(timeout_us)) {
                return std::unexpected{std::monostate{}};
            }
            return value;
        }

        bool try_store_for(T new_value, std::chrono::microseconds timeout_us) {
            std::unique_lock lock{mtx, std::defer_lock};
            if(!lock.try_lock_for(timeout_us)) {
                return false;
            }
            value = std::move(new_value);
            return true;
        }
};

#endif