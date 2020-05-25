#pragma once

#include <iostream>
#include <inference_engine.hpp>

namespace space
{

#if 0

#pragma region ObjectDetectionBase
	/// <summary>
	/// 对象检测基类
	/// </summary>
	class ObjectDetectionBase
	{
	public:
		/// <summary>
		/// 基本对象检测构造函数
		/// </summary>
		/// <param name="modelPath">AI模型路径</param>
		/// <param name="device">推断使用的硬件类型</param>
		/// <param name="isAsync">是否异步执行</param>
		/// <returns></returns>
		ObjectDetectionBase(const std::string& modelPath, const std::string* device, bool isAsync);
		~ObjectDetectionBase();

		/// <summary>
		/// 可执行网络
		/// </summary>
		/// <returns></returns>
		InferenceEngine::ExecutableNetwork* operator ->();

		/// <summary>
		/// 读取网络
		/// </summary>
		/// <param name="ie"></param>
		/// <returns></returns>
		virtual InferenceEngine::CNNNetwork read(const InferenceEngine::Core& ie) = 0;
		
		/// <summary>
		/// 提交推断请求
		/// </summary>
		virtual void submitRequest();

		/// <summary>
		/// 异步等待
		/// </summary>
		virtual void wait();

		/// <summary>
		/// 该对应检测是否激活可用
		/// </summary>
		/// <returns></returns>
		bool enabled() const;

	protected:
		std::string network_input_name;
		std::string network_output_name;
	};

#pragma endregion

#endif

}
