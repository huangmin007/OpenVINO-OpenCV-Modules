#pragma once

#include <open_model_infer.hpp>

namespace space
{
    class ObjectRecognition : public OpenModelInferBase
    {
    public:
		ObjectRecognition(const std::vector<std::string>& output_layers_name, bool is_debug = true);
		~ObjectRecognition();

		void SetParentNetwork(OpenModelInferBase* parent);

	protected:
		void ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status) override;
		void UpdateDebugShow() override;

	private:
    };
}
