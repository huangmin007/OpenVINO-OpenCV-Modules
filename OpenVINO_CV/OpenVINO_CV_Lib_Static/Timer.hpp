#pragma once

/*
Timer timer;

// execute task every timer interval
std::cout << "--- start period timer ----" << std::endl;
timer.start(1000, std::bind(func2, 3));
std::this_thread::sleep_for(std::chrono::milliseconds(5000));
timer.stop();
std::cout << "--- stop period timer ----" << std::endl;

// execute task once after delay
std::cout << "--- start one shot timer ----" << std::endl;
timer.startOnce(1000, func1);
std::cout << "--- stop one shot timer ----" << std::endl;

*/

#include <functional>
#include <chrono>
#include <thread>
#include <atomic>
#include <memory>
#include <mutex>
#include <condition_variable>

namespace space
{
	class Timer
	{
	public:
		Timer() : _expired(true), _try_to_expire(false)
		{}

		Timer(const Timer& timer)
		{
			_expired = timer._expired.load();
			_try_to_expire = timer._try_to_expire.load();
		}

		~Timer()
		{
			stop();
		}

		void start(int interval, std::function<void()> task)
		{
			// 已启动，请勿再次启动
			if (_expired == false)
				return;

			// 启动异步计时器，启动线程并在该线程中等待
			_expired = false;
			std::thread([this, interval, task]()
				{
					while (!_try_to_expire)
					{
						std::this_thread::sleep_for(std::chrono::milliseconds(interval));
						task();
					}

					{
						//计时器停止运行，更新条件变量过期并唤醒主线程
						std::lock_guard<std::mutex> locker(_mutex);
						_expired = true;
						_expired_cond.notify_one();
					}
				}).detach();
		}

		void startOnce(int delay, std::function<void()> task)
		{
			std::thread([delay, task]()
				{
					std::this_thread::sleep_for(std::chrono::milliseconds(delay));
					task();
				}).detach();
		}

		void stop()
		{
			if (_expired) return;

			if (_try_to_expire) return;

			_try_to_expire = true;
			{
				std::unique_lock<std::mutex> locker(_mutex);
				_expired_cond.wait(locker, [this] {return _expired == true; });

				if (_expired == true) _try_to_expire = false;
			}
		}

	private:
		std::atomic<bool> _expired; //计时器停止状态
		std::atomic<bool> _try_to_expire; // 计时器正在停止
		std::mutex _mutex;
		std::condition_variable _expired_cond;
	};
}