#pragma once

#include <iostream>
#include <vector>
#include <shared_mutex>
#include <Windows.h>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

#include "timer.hpp"
#include "open_model_infer.hpp"

namespace space
{
	/// <summary>
	/// Áª¼¶ÍÆ¶Ï
	/// </summary>
	class OpenModelMultiInfer :public OpenModelInferBase
	{
	public:
		OpenModelMultiInfer(
			const std::vector<std::string>& output_layers, 
			const std::vector<std::vector<std::string>>& sub_output_layers,
			bool is_debug = true);
		~OpenModelMultiInfer();

		void ConfigSubNetwork(InferenceEngine::Core& ie, const std::vector<std::string>& sub_model_info);

	protected:
		void ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode code) override;
		void UpdateDebugShow() override;

		void SetSubNetworkIO();
		void SetSubInferCallback();
		void SetSubMemoryShared();

		std::vector<InferenceEngine::CNNNetwork> sub_cnnNetworks;
		std::vector<InferenceEngine::CNNNetwork> sub_execNetworks;

		std::vector<std::vector<std::pair<std::string, LPVOID>>> sub_shared_layers;

	private:
		std::vector<std::vector<std::string>> sub_output_layers;
		std::vector<std::vector<std::pair<std::string, std::size_t>>> sub_shared_layers_info;

		std::vector<std::vector<HANDLE>> sub_shared_handle;
	};
}

