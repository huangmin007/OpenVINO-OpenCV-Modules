#include "pch.h"
#include "open_model_multi_infer.hpp"
#include "static_functions.hpp"

namespace space
{
	OpenModelMultiInfer::OpenModelMultiInfer(const std::vector<std::string>& output_layers,
		const std::vector<std::vector<std::string>>& sub_output_layers, bool is_debug)
		:OpenModelInferBase(output_layers, is_debug), sub_output_layers(sub_output_layers)
	{
	}

	OpenModelMultiInfer::~OpenModelMultiInfer()
	{
		OpenModelInferBase::~OpenModelInferBase();
		
		LOG("INFO") << "OpenModelMultiInfer 正在关闭/清理共享内存  ... " << std::endl;
		for (auto& vec : sub_shared_layers)
		{
			for (auto& shared : vec)
			{
				UnmapViewOfFile(shared.second);
				shared.second = NULL;
			}

			vec.clear();
		}

		for (auto& vec : sub_shared_handle)
		{
			for (auto& handle : vec)
			{
				CloseHandle(handle);
				handle = NULL;
			}
			vec.clear();
		}

		sub_shared_layers.clear();
		sub_shared_handle.clear();
	}

	void OpenModelMultiInfer::ConfigSubNetwork(InferenceEngine::Core& ie, const std::vector<std::string>& sub_model_info)
	{

	}

	void OpenModelMultiInfer::SetSubNetworkIO()
	{

	}
	void OpenModelMultiInfer::SetSubInferCallback()
	{

	}
	void OpenModelMultiInfer::SetSubMemoryShared()
	{

	}

	void OpenModelMultiInfer::ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode code)
	{

	}
	void OpenModelMultiInfer::UpdateDebugShow()
	{

	}
}