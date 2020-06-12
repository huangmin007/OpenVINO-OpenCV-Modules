#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include "actions.hpp"
#include <samples\common.hpp>
#include "tracker.hpp"
#include "logger.hpp"
#include "face_reid.hpp"
#include <static_functions.hpp>

//using namespace InferenceEngine;

namespace {

    class Visualizer {
    private:
        cv::Mat frame_;
        cv::Mat top_persons_;
        const bool enabled_;
        const int num_top_persons_;
        cv::VideoWriter& writer_;
        float rect_scale_x_;
        float rect_scale_y_;
        static int const max_input_width_ = 1920;
        std::string const main_window_name_ = "Smart classroom demo";
        std::string const top_window_name_ = "Top-k students";
        static int const crop_width_ = 128;
        static int const crop_height_ = 320;
        static int const header_size_ = 80;
        static int const margin_size_ = 5;

    public:
        Visualizer(bool enabled, cv::VideoWriter& writer, int num_top_persons) : enabled_(enabled), num_top_persons_(num_top_persons), writer_(writer),
            rect_scale_x_(0), rect_scale_y_(0) {
            if (!enabled_) {
                return;
            }

            cv::namedWindow(main_window_name_);

            if (num_top_persons_ > 0) {
                cv::namedWindow(top_window_name_);

                CreateTopWindow();
                ClearTopWindow();
            }
        }

        static cv::Size GetOutputSize(const cv::Size& input_size) {
            if (input_size.width > max_input_width_) {
                float ratio = static_cast<float>(input_size.height) / input_size.width;
                return cv::Size(max_input_width_, cvRound(ratio * max_input_width_));
            }
            return input_size;
        }

        void SetFrame(const cv::Mat& frame) {
            if (!enabled_ && !writer_.isOpened()) {
                return;
            }

            frame_ = frame.clone();
            rect_scale_x_ = 1;
            rect_scale_y_ = 1;
            cv::Size new_size = GetOutputSize(frame_.size());
            if (new_size != frame_.size()) {
                rect_scale_x_ = static_cast<float>(new_size.height) / frame_.size().height;
                rect_scale_y_ = static_cast<float>(new_size.width) / frame_.size().width;
                cv::resize(frame_, frame_, new_size);
            }
        }

        void Show() const {
            if (enabled_) {
                cv::imshow(main_window_name_, frame_);
            }

            if (writer_.isOpened()) {
                writer_ << frame_;
            }
        }

        void DrawCrop(cv::Rect roi, int id, const cv::Scalar& color) const {
            if (!enabled_ || num_top_persons_ <= 0) {
                return;
            }

            if (id < 0 || id >= num_top_persons_) {
                return;
            }

            if (rect_scale_x_ != 1 || rect_scale_y_ != 1) {
                roi.x = cvRound(roi.x * rect_scale_x_);
                roi.y = cvRound(roi.y * rect_scale_y_);

                roi.height = cvRound(roi.height * rect_scale_y_);
                roi.width = cvRound(roi.width * rect_scale_x_);
            }

            roi.x = std::max(0, roi.x);
            roi.y = std::max(0, roi.y);
            roi.width = std::min(roi.width, frame_.cols - roi.x);
            roi.height = std::min(roi.height, frame_.rows - roi.y);

            const auto crop_label = std::to_string(id + 1);

            auto frame_crop = frame_(roi).clone();
            cv::resize(frame_crop, frame_crop, cv::Size(crop_width_, crop_height_));

            const int shift = (id + 1) * margin_size_ + id * crop_width_;
            frame_crop.copyTo(top_persons_(cv::Rect(shift, header_size_, crop_width_, crop_height_)));

            cv::imshow(top_window_name_, top_persons_);
        }

        void DrawObject(cv::Rect rect, const std::string& label_to_draw,
            const cv::Scalar& text_color, const cv::Scalar& bbox_color, bool plot_bg) {
            if (!enabled_ && !writer_.isOpened()) {
                return;
            }

            if (rect_scale_x_ != 1 || rect_scale_y_ != 1) {
                rect.x = cvRound(rect.x * rect_scale_x_);
                rect.y = cvRound(rect.y * rect_scale_y_);

                rect.height = cvRound(rect.height * rect_scale_y_);
                rect.width = cvRound(rect.width * rect_scale_x_);
            }
            cv::rectangle(frame_, rect, bbox_color);

            if (plot_bg && !label_to_draw.empty()) {
                int baseLine = 0;
                const cv::Size label_size =
                    cv::getTextSize(label_to_draw, cv::FONT_HERSHEY_PLAIN, 1, 1, &baseLine);
                cv::rectangle(frame_, cv::Point(rect.x, rect.y - label_size.height),
                    cv::Point(rect.x + label_size.width, rect.y + baseLine),
                    bbox_color, cv::FILLED);
            }
            if (!label_to_draw.empty()) {
                cv::putText(frame_, label_to_draw, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_PLAIN, 1,
                    text_color, 1, cv::LINE_AA);
            }
        }

        void DrawFPS(const float fps, const cv::Scalar& color) {
            if (enabled_ && !writer_.isOpened()) {
                cv::putText(frame_,
                    std::to_string(static_cast<int>(fps)) + " fps",
                    cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1,
                    color, 2, cv::LINE_AA);
            }
        }

        void CreateTopWindow() {
            if (!enabled_ || num_top_persons_ <= 0) {
                return;
            }

            const int width = margin_size_ * (num_top_persons_ + 1) + crop_width_ * num_top_persons_;
            const int height = header_size_ + crop_height_ + margin_size_;

            top_persons_.create(height, width, CV_8UC3);
        }

        void ClearTopWindow() {
            if (!enabled_ || num_top_persons_ <= 0) {
                return;
            }

            top_persons_.setTo(cv::Scalar(255, 255, 255));

            for (int i = 0; i < num_top_persons_; ++i) {
                const int shift = (i + 1) * margin_size_ + i * crop_width_;

                cv::rectangle(top_persons_, cv::Point(shift, header_size_),
                    cv::Point(shift + crop_width_, header_size_ + crop_height_),
                    cv::Scalar(128, 128, 128), cv::FILLED);

                const auto label_to_draw = "#" + std::to_string(i + 1);
                int baseLine = 0;
                const auto label_size =
                    cv::getTextSize(label_to_draw, cv::FONT_HERSHEY_SIMPLEX, 2, 2, &baseLine);
                const int text_shift = (crop_width_ - label_size.width) / 2;
                cv::putText(top_persons_, label_to_draw,
                    cv::Point(shift + text_shift, label_size.height + baseLine / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            }

            cv::imshow(top_window_name_, top_persons_);
        }

        void Finalize() const {
            if (enabled_) {
                cv::destroyWindow(main_window_name_);

                if (num_top_persons_ > 0) {
                    cv::destroyWindow(top_window_name_);
                }
            }

            if (writer_.isOpened()) {
                writer_.release();
            }
        }
    };

    const int default_action_index = -1;  // Unknown action class

    void ConvertActionMapsToFrameEventTracks(const std::vector<std::map<int, int>>& obj_id_to_action_maps,
        int default_action,
        std::map<int, FrameEventsTrack>* obj_id_to_actions_track) {
        for (size_t frame_id = 0; frame_id < obj_id_to_action_maps.size(); ++frame_id) {
            for (const auto& tup : obj_id_to_action_maps[frame_id]) {
                if (tup.second != default_action) {
                    (*obj_id_to_actions_track)[tup.first].emplace_back(frame_id, tup.second);
                }
            }
        }
    }

    void SmoothTracks(const std::map<int, FrameEventsTrack>& obj_id_to_actions_track,
        int start_frame, int end_frame, int window_size, int min_length, int default_action,
        std::map<int, RangeEventsTrack>* obj_id_to_events) {
        // Iterate over face tracks
        for (const auto& tup : obj_id_to_actions_track) {
            const auto& frame_events = tup.second;
            if (frame_events.empty()) {
                continue;
            }

            RangeEventsTrack range_events;


            // Merge neighbouring events and filter short ones
            range_events.emplace_back(frame_events.front().frame_id,
                frame_events.front().frame_id + 1,
                frame_events.front().action);

            for (size_t frame_id = 1; frame_id < frame_events.size(); ++frame_id) {
                const auto& last_range_event = range_events.back();
                const auto& cur_frame_event = frame_events[frame_id];

                if (last_range_event.end_frame_id + window_size - 1 >= cur_frame_event.frame_id &&
                    last_range_event.action == cur_frame_event.action) {
                    range_events.back().end_frame_id = cur_frame_event.frame_id + 1;
                }
                else {
                    if (range_events.back().end_frame_id - range_events.back().begin_frame_id < min_length) {
                        range_events.pop_back();
                    }

                    range_events.emplace_back(cur_frame_event.frame_id,
                        cur_frame_event.frame_id + 1,
                        cur_frame_event.action);
                }
            }
            if (range_events.back().end_frame_id - range_events.back().begin_frame_id < min_length) {
                range_events.pop_back();
            }

            // Extrapolate track
            if (range_events.empty()) {
                range_events.emplace_back(start_frame, end_frame, default_action);
            }
            else {
                range_events.front().begin_frame_id = start_frame;
                range_events.back().end_frame_id = end_frame;
            }

            // Interpolate track
            for (size_t event_id = 1; event_id < range_events.size(); ++event_id) {
                auto& last_event = range_events[event_id - 1];
                auto& cur_event = range_events[event_id];

                int middle_point = static_cast<int>(0.5f * (cur_event.begin_frame_id + last_event.end_frame_id));

                cur_event.begin_frame_id = middle_point;
                last_event.end_frame_id = middle_point;
            }

            // Merge consecutive events
            auto& final_events = (*obj_id_to_events)[tup.first];
            final_events.push_back(range_events.front());
            for (size_t event_id = 1; event_id < range_events.size(); ++event_id) {
                const auto& cur_event = range_events[event_id];

                if (final_events.back().action == cur_event.action) {
                    final_events.back().end_frame_id = cur_event.end_frame_id;
                }
                else {
                    final_events.push_back(cur_event);
                }
            }
        }
    }

    void ConvertRangeEventsTracksToActionMaps(int num_frames,
        const std::map<int, RangeEventsTrack>& obj_id_to_events,
        std::vector<std::map<int, int>>* obj_id_to_action_maps) {
        obj_id_to_action_maps->resize(num_frames);

        for (const auto& tup : obj_id_to_events) {
            const int obj_id = tup.first;
            const auto& events = tup.second;

            for (const auto& event : events) {
                for (int frame_id = event.begin_frame_id; frame_id < event.end_frame_id; ++frame_id) {
                    (*obj_id_to_action_maps)[frame_id].emplace(obj_id, event.action);
                }
            }
        }
    }

    std::vector<std::string> ParseActionLabels(const std::string& in_str) {
        std::vector<std::string> labels;
        std::string label;
        std::istringstream stream_to_split(in_str);

        while (std::getline(stream_to_split, label, ',')) {
            labels.push_back(label);
        }

        return labels;
    }

    std::string GetActionTextLabel(const unsigned label, const std::vector<std::string>& actions_map) {
        if (label < actions_map.size()) {
            return actions_map[label];
        }
        return "__undefined__";
    }

    cv::Scalar GetActionTextColor(const unsigned label) {
        static const cv::Scalar label_colors[] = {
            cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 255) };
        if (label < arraySize(label_colors)) {
            return label_colors[label];
        }
        return cv::Scalar(0, 0, 0);
    }

    float CalculateIoM(const cv::Rect& rect1, const cv::Rect& rect2) {
        int area1 = rect1.area();
        int area2 = rect2.area();

        float area_min = static_cast<float>(std::min(area1, area2));
        float area_intersect = static_cast<float>((rect1 & rect2).area());

        return area_intersect / area_min;
    }

    cv::Rect DecreaseRectByRelBorders(const cv::Rect& r) {
        float w = static_cast<float>(r.width);
        float h = static_cast<float>(r.height);

        float left = std::ceil(w * 0.0f);
        float top = std::ceil(h * 0.0f);
        float right = std::ceil(w * 0.0f);
        float bottom = std::ceil(h * .7f);

        cv::Rect res;
        res.x = r.x + static_cast<int>(left);
        res.y = r.y + static_cast<int>(top);
        res.width = static_cast<int>(r.width - left - right);
        res.height = static_cast<int>(r.height - top - bottom);
        return res;
    }

    int GetIndexOfTheNearestPerson(const TrackedObject& face, const std::vector<TrackedObject>& tracked_persons) {
        int argmax = -1;
        float max_iom = std::numeric_limits<float>::lowest();
        for (size_t i = 0; i < tracked_persons.size(); i++) {
            float iom = CalculateIoM(face.rect, DecreaseRectByRelBorders(tracked_persons[i].rect));
            if ((iom > 0) && (iom > max_iom)) {
                max_iom = iom;
                argmax = i;
            }
        }
        return argmax;
    }

    std::map<int, int> GetMapFaceTrackIdToLabel(const std::vector<Track>& face_tracks) {
        std::map<int, int> face_track_id_to_label;
        for (const auto& track : face_tracks) {
            const auto& first_obj = track.first_object;
            // check consistency
            // to receive this consistency for labels
            // use the function UpdateTrackLabelsToBestAndFilterOutUnknowns
            for (const auto& obj : track.objects) {
                SCR_CHECK_EQ(obj.label, first_obj.label);
                SCR_CHECK_EQ(obj.object_id, first_obj.object_id);
            }

            auto cur_obj_id = first_obj.object_id;
            auto cur_label = first_obj.label;
            SCR_CHECK(face_track_id_to_label.count(cur_obj_id) == 0) << " Repeating face tracks";
            face_track_id_to_label[cur_obj_id] = cur_label;
        }
        return face_track_id_to_label;
    }

    bool checkDynamicBatchSupport(const InferenceEngine::Core& ie, const std::string& device) {
        try {
            if (ie.GetConfig(device, CONFIG_KEY(DYN_BATCH_ENABLED)).as<std::string>() != InferenceEngine::PluginConfigParams::YES)
                return false;
        }
        catch (const std::exception&) {
            return false;
        }
        return true;
    }

    class FaceRecognizer {
    public:
        virtual ~FaceRecognizer() = default;

        virtual bool LabelExists(const std::string& label) const = 0;
        virtual std::string GetLabelByID(int id) const = 0;
        virtual std::vector<std::string> GetIDToLabelMap() const = 0;

        virtual std::vector<int> Recognize(const cv::Mat& frame, const detection::DetectedObjects& faces) = 0;

        virtual void PrintPerformanceCounts(
            const std::string& landmarks_device, const std::string& reid_device) = 0;
    };

    class FaceRecognizerNull : public FaceRecognizer {
    public:
        bool LabelExists(const std::string&) const override { return false; }

        std::string GetLabelByID(int) const override {
            return EmbeddingsGallery::unknown_label;
        }

        std::vector<std::string> GetIDToLabelMap() const override { return {}; }

        std::vector<int> Recognize(const cv::Mat&, const detection::DetectedObjects& faces) override {
            return std::vector<int>(faces.size(), EmbeddingsGallery::unknown_id);
        }

        void PrintPerformanceCounts(
            const std::string&, const std::string&) override {}
    };

    class FaceRecognizerDefault : public FaceRecognizer {
    public:
        FaceRecognizerDefault(
            const CnnConfig& landmarks_detector_config,
            const CnnConfig& reid_config,
            const detection::DetectorConfig& face_registration_det_config,
            const std::string& face_gallery_path,
            double reid_threshold,
            int min_size_fr,
            bool crop_gallery,
            bool greedy_reid_matching
        )
            : landmarks_detector(landmarks_detector_config),
            face_reid(reid_config),
            face_gallery(face_gallery_path, reid_threshold, min_size_fr, crop_gallery,
                face_registration_det_config, landmarks_detector, face_reid,
                greedy_reid_matching)
        {
            if (face_gallery.size() == 0) {
                LOG("WARN") << "Face reid gallery is empty!" << std::endl;
            }
            else {
                LOG("INFO") << "Face reid gallery size: " << face_gallery.size() << std::endl;
            }
        }

        bool LabelExists(const std::string& label) const override {
            return face_gallery.LabelExists(label);
        }

        std::string GetLabelByID(int id) const override {
            return face_gallery.GetLabelByID(id);
        }

        std::vector<std::string> GetIDToLabelMap() const override {
            return face_gallery.GetIDToLabelMap();
        }

        std::vector<int> Recognize(const cv::Mat& frame, const detection::DetectedObjects& faces) override {
            std::vector<cv::Mat> face_rois;

            for (const auto& face : faces) {
                face_rois.push_back(frame(face.rect));
            }

            std::vector<cv::Mat> landmarks, embeddings;

            landmarks_detector.Compute(face_rois, &landmarks, cv::Size(2, 5));
            AlignFaces(&face_rois, &landmarks);
            face_reid.Compute(face_rois, &embeddings);
            return face_gallery.GetIDsByEmbeddings(embeddings);
        }

        void PrintPerformanceCounts(
            const std::string& landmarks_device, const std::string& reid_device) {
            landmarks_detector.PrintPerformanceCounts(landmarks_device);
            face_reid.PrintPerformanceCounts(reid_device);
        }

    private:
        VectorCNN landmarks_detector;
        VectorCNN face_reid;
        EmbeddingsGallery face_gallery;
    };

}  // namespace
