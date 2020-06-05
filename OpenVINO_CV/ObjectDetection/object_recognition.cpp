#include "object_recognition.hpp"

namespace space
{
	ObjectRecognition::ObjectRecognition(const std::vector<std::string>& output_layers_name, bool is_debug)
		:OpenModelInferBase(output_layers_name, is_debug) {};

	ObjectRecognition::~ObjectRecognition() {};

	void ObjectRecognition::SetParentNetwork(OpenModelInferBase* parent)
	{
		if (parent == this)
			throw std::logic_error("父级网络不可设置为自已本身 ... ");

		parent->SetCompletionCallback(
			[&](InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status)
			{
				std::cout << "Parent ... " << std::endl;
			}
		);
	}


	void ObjectRecognition::ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status)
	{
	}

	void ObjectRecognition::UpdateDebugShow()
	{
	}
}