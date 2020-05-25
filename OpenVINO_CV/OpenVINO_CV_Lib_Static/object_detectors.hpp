#pragma once

#include <iostream>
#include <inference_engine.hpp>

namespace space
{

#if 0

#pragma region ObjectDetectionBase
	/// <summary>
	/// ���������
	/// </summary>
	class ObjectDetectionBase
	{
	public:
		/// <summary>
		/// ���������⹹�캯��
		/// </summary>
		/// <param name="modelPath">AIģ��·��</param>
		/// <param name="device">�ƶ�ʹ�õ�Ӳ������</param>
		/// <param name="isAsync">�Ƿ��첽ִ��</param>
		/// <returns></returns>
		ObjectDetectionBase(const std::string& modelPath, const std::string* device, bool isAsync);
		~ObjectDetectionBase();

		/// <summary>
		/// ��ִ������
		/// </summary>
		/// <returns></returns>
		InferenceEngine::ExecutableNetwork* operator ->();

		/// <summary>
		/// ��ȡ����
		/// </summary>
		/// <param name="ie"></param>
		/// <returns></returns>
		virtual InferenceEngine::CNNNetwork read(const InferenceEngine::Core& ie) = 0;
		
		/// <summary>
		/// �ύ�ƶ�����
		/// </summary>
		virtual void submitRequest();

		/// <summary>
		/// �첽�ȴ�
		/// </summary>
		virtual void wait();

		/// <summary>
		/// �ö�Ӧ����Ƿ񼤻����
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
