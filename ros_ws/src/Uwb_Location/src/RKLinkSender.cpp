#include <ros/ros.h>
#include <serial/serial.h>

#include <Uwb_Location/uwb.h>
#include <animal_vision/DetectionEvent.h>
#include <animal_vision/K230Detection.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>


namespace rk_link
{
constexpr uint8_t kYoloPayloadVersion = 0x02;
constexpr uint8_t kFrameHead1 = 0xAA;
constexpr uint8_t kFrameHead2 = 0x55;
constexpr uint8_t kVersion = 0x01;
constexpr uint8_t kMsgUwb = 0x01;
constexpr uint8_t kMsgYolo = 0x02;
constexpr size_t kMaxPayload = 32;
constexpr uint8_t kK230PayloadVersion = 0x01;
constexpr uint8_t kMsgK230 = 0x03;

uint16_t crc16Modbus(const uint8_t* data, size_t len)
{
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; ++i)
    {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit)
        {
            if ((crc & 0x0001U) != 0U)
            {
                crc = static_cast<uint16_t>((crc >> 1U) ^ 0xA001U);
            }
            else
            {
                crc = static_cast<uint16_t>(crc >> 1U);
            }
        }
    }
    return crc;
}

void appendU8(std::vector<uint8_t>& out, uint8_t value)
{
    out.push_back(value);
}

void appendU16Le(std::vector<uint8_t>& out, uint16_t value)
{
    out.push_back(static_cast<uint8_t>(value & 0xFFU));
    out.push_back(static_cast<uint8_t>((value >> 8U) & 0xFFU));
}

void appendU32Le(std::vector<uint8_t>& out, uint32_t value)
{
    out.push_back(static_cast<uint8_t>(value & 0xFFU));
    out.push_back(static_cast<uint8_t>((value >> 8U) & 0xFFU));
    out.push_back(static_cast<uint8_t>((value >> 16U) & 0xFFU));
    out.push_back(static_cast<uint8_t>((value >> 24U) & 0xFFU));
}

void appendI16Le(std::vector<uint8_t>& out, int16_t value)
{
    appendU16Le(out, static_cast<uint16_t>(value));
}

void appendI32Le(std::vector<uint8_t>& out, int32_t value)
{
    const uint32_t raw = static_cast<uint32_t>(value);
    out.push_back(static_cast<uint8_t>(raw & 0xFFU));
    out.push_back(static_cast<uint8_t>((raw >> 8U) & 0xFFU));
    out.push_back(static_cast<uint8_t>((raw >> 16U) & 0xFFU));
    out.push_back(static_cast<uint8_t>((raw >> 24U) & 0xFFU));
}

std::vector<uint8_t> packFrame(
    uint8_t msg_id,
    uint8_t seq,
    const std::vector<uint8_t>& payload)
{
    if (payload.size() > kMaxPayload)
    {
        throw std::runtime_error("RK_LINK payload too long");
    }

    std::vector<uint8_t> body;
    body.reserve(4U + payload.size());
    appendU8(body, kVersion);
    appendU8(body, msg_id);
    appendU8(body, seq);
    appendU8(body, static_cast<uint8_t>(payload.size()));
    body.insert(body.end(), payload.begin(), payload.end());

    const uint16_t crc = crc16Modbus(body.data(), body.size());

    std::vector<uint8_t> frame;
    frame.reserve(2U + body.size() + 2U);
    appendU8(frame, kFrameHead1);
    appendU8(frame, kFrameHead2);
    frame.insert(frame.end(), body.begin(), body.end());
    appendU16Le(frame, crc);
    return frame;
}

int32_t metersToCentimeters(float value_m)
{
    return static_cast<int32_t>(
        std::llround(static_cast<double>(value_m) * 100.0));
}



}  // namespace rk_link

class RKLinkSender
{
public:
    RKLinkSender()
        : nh_(),
          pnh_("~"),
          baudrate_(115200),
          send_rate_hz_(20.0),
          uwb_history_sec_(5.0),
          vision_uwb_max_delta_ms_(150.0),
          coordinate_abs_limit_cm_(1000),
          max_history_samples_(500),
          seq_(0),
          detection_latched_(false),
          latched_event_id_(0),
          latched_first_seen_(),
          latched_x_cm_(0),
          latched_y_cm_(0)
    {
        pnh_.param<std::string>(
            "port", port_name_, "/dev/serial/by-id/fc_serial");
        pnh_.param("baudrate", baudrate_, 115200);
        pnh_.param<std::string>("uwb_topic", uwb_topic_, "/uwb/data");
        pnh_.param<std::string>(
            "vision_event_topic",
            vision_event_topic_,
            "/vision/detection_event");
        pnh_.param<std::string>(
            "k230_topic",
            k230_topic_,
            "/k230/detection");
        pnh_.param("send_rate", send_rate_hz_, 20.0);
        pnh_.param("uwb_history_sec", uwb_history_sec_, 5.0);
        pnh_.param(
            "vision_uwb_max_delta_ms",
            vision_uwb_max_delta_ms_,
            150.0);
        pnh_.param(
            "coordinate_abs_limit_cm",
            coordinate_abs_limit_cm_,
            1000);
        pnh_.param("max_history_samples", max_history_samples_, 500);

        serial_.setPort(port_name_);
        serial_.setBaudrate(static_cast<uint32_t>(baudrate_));
        serial::Timeout serial_timeout =
        serial::Timeout::simpleTimeout(20);
        serial_.setTimeout(serial_timeout);

        uwb_sub_ = nh_.subscribe(
            uwb_topic_, 50, &RKLinkSender::onUwb, this);
        vision_sub_ = nh_.subscribe(
            vision_event_topic_, 10, &RKLinkSender::onVision, this);

        k230_sub_ = nh_.subscribe(
            k230_topic_,
            10,
            &RKLinkSender::onK230,
            this);

        reconnect_timer_ = nh_.createTimer(
            ros::Duration(1.0),
            &RKLinkSender::onReconnectTimer,
            this);

        openSerial();

        ROS_INFO_STREAM(
            "RKLinkSender configured port=" << port_name_
            << " baudrate=" << baudrate_
            << " uwb_topic=" << uwb_topic_
            << " vision_event_topic=" << vision_event_topic_
            << " k230_topic=" << k230_topic_
            << " send_rate=" << send_rate_hz_
            << " history_sec=" << uwb_history_sec_
            << " max_match_delta_ms=" << vision_uwb_max_delta_ms_);
            
    }

private:
    struct UwbSample
    {
        ros::Time stamp;
        int32_t x_cm;
        int32_t y_cm;
    };

    bool openSerial()
    {
        if (serial_.isOpen())
        {
            return true;
        }

        try
        {
            serial_.open();
            if (serial_.isOpen())
            {
                ROS_INFO_STREAM(
                    "RK_LINK serial opened: " << port_name_
                    << " baudrate=" << baudrate_);
                return true;
            }
        }
        catch (const std::exception& error)
        {
            ROS_WARN_THROTTLE(
                2.0,
                "RK_LINK serial open failed: %s",
                error.what());
        }
        return false;
    }

    void closeSerial()
    {
        try
        {
            if (serial_.isOpen())
            {
                serial_.close();
            }
        }
        catch (const std::exception& error)
        {
            ROS_WARN_THROTTLE(
                2.0,
                "RK_LINK serial close failed: %s",
                error.what());
        }
    }

    void onReconnectTimer(const ros::TimerEvent&)
    {
        if (!serial_.isOpen())
        {
            openSerial();
        }
    }

    bool writeFrame(const std::vector<uint8_t>& frame)
    {
        if (!serial_.isOpen())
        {
            ROS_WARN_THROTTLE(
                1.0, "RK_LINK serial offline; frame is not sent");
            return false;
        }

        try
        {
            const size_t written = serial_.write(frame);
            if (written != frame.size())
            {
                ROS_WARN_THROTTLE(
                    1.0,
                    "RK_LINK short write: %zu/%zu",
                    written,
                    frame.size());
                return false;
            }
            return true;
        }
        catch (const std::exception& error)
        {
            ROS_ERROR_THROTTLE(
                1.0,
                "RK_LINK serial write failed: %s",
                error.what());
            closeSerial();
            return false;
        }
    }

    void storeUwbSample(
        const ros::Time& stamp,
        int32_t x_cm,
        int32_t y_cm)
    {
        if (!uwb_history_.empty() && stamp < uwb_history_.back().stamp)
        {
            ROS_WARN(
                "ROS time moved backwards; clearing UWB history");
            uwb_history_.clear();
        }

        UwbSample sample;
        sample.stamp = stamp;
        sample.x_cm = x_cm;
        sample.y_cm = y_cm;
        uwb_history_.push_back(sample);

        while (!uwb_history_.empty())
        {
            const double age_sec =
                (stamp - uwb_history_.front().stamp).toSec();
            if (age_sec <= uwb_history_sec_ &&
                static_cast<int>(uwb_history_.size()) <=
                    max_history_samples_)
            {
                break;
            }
            uwb_history_.pop_front();
        }
    }

    bool validDetectionCoordinate(int32_t x_cm, int32_t y_cm) const
    {
        const int32_t i16_min =
            static_cast<int32_t>(std::numeric_limits<int16_t>::min());
        const int32_t i16_max =
            static_cast<int32_t>(std::numeric_limits<int16_t>::max());

        return x_cm >= i16_min && x_cm <= i16_max &&
               y_cm >= i16_min && y_cm <= i16_max &&
               std::llabs(static_cast<long long>(x_cm)) <=
                   static_cast<long long>(coordinate_abs_limit_cm_) &&
               std::llabs(static_cast<long long>(y_cm)) <=
                   static_cast<long long>(coordinate_abs_limit_cm_);
    }

    bool findNearestUwb(
        const ros::Time& target_stamp,
        int16_t* x_cm,
        int16_t* y_cm,
        double* delta_ms) const
    {
        if (target_stamp.isZero() || uwb_history_.empty())
        {
            return false;
        }

        const UwbSample* nearest = nullptr;
        double nearest_delta_ms =
            std::numeric_limits<double>::infinity();

        for (const UwbSample& sample : uwb_history_)
        {
            const double candidate_delta_ms =
                std::fabs((sample.stamp - target_stamp).toSec()) * 1000.0;
            if (candidate_delta_ms < nearest_delta_ms)
            {
                nearest = &sample;
                nearest_delta_ms = candidate_delta_ms;
            }
        }

        if (nearest == nullptr ||
            nearest_delta_ms > vision_uwb_max_delta_ms_ ||
            !validDetectionCoordinate(nearest->x_cm, nearest->y_cm))
        {
            return false;
        }

        *x_cm = static_cast<int16_t>(nearest->x_cm);
        *y_cm = static_cast<int16_t>(nearest->y_cm);
        *delta_ms = nearest_delta_ms;
        return true;
    }

        void sendYolo(
        bool target_found,
        uint8_t class_id,
        uint8_t animal_count,
        int16_t x_cm,
        int16_t y_cm,
        uint32_t event_id)
    {
        std::vector<uint8_t> payload;
        payload.reserve(12);

        rk_link::appendU8(
            payload,
            target_found ? 1U : 0U);
        rk_link::appendU8(payload, class_id);
        rk_link::appendU8(payload, animal_count);
        rk_link::appendU8(
            payload,
            rk_link::kYoloPayloadVersion);
        rk_link::appendI16Le(payload, x_cm);
        rk_link::appendI16Le(payload, y_cm);
        rk_link::appendU32Le(payload, event_id);

        writeFrame(
            rk_link::packFrame(
                rk_link::kMsgYolo,
                nextSeq(),
                payload));
    }
    void onUwb(const Uwb_Location::uwb::ConstPtr& message)
    {
        const ros::Time stamp = message->header.stamp.isZero()
            ? ros::Time::now()
            : message->header.stamp;
        const int32_t x_cm =
            rk_link::metersToCentimeters(message->x);
        const int32_t y_cm =
            rk_link::metersToCentimeters(message->y);

        // Every UWB sample enters the history before serial rate limiting.
        storeUwbSample(stamp, x_cm, y_cm);

        const ros::Time now = ros::Time::now();
        if (send_rate_hz_ > 0.0 && !last_uwb_send_.isZero())
        {
            const ros::Duration min_period(1.0 / send_rate_hz_);
            if ((now - last_uwb_send_) < min_period)
            {
                return;
            }
        }
        last_uwb_send_ = now;

        std::vector<uint8_t> payload;
        payload.reserve(16);
        rk_link::appendI32Le(payload, x_cm);
        rk_link::appendI32Le(payload, y_cm);
        rk_link::appendI16Le(payload, 0);
        rk_link::appendI16Le(payload, 0);
        rk_link::appendU16Le(payload, 100);
        rk_link::appendU16Le(payload, 0);

        writeFrame(
            rk_link::packFrame(
                rk_link::kMsgUwb,
                nextSeq(),
                payload));

        ROS_INFO_THROTTLE(
            1.0,
            "RK_LINK UWB x_cm=%d y_cm=%d history=%zu",
            static_cast<int>(x_cm),
            static_cast<int>(y_cm),
            uwb_history_.size());
    }

    void onVision(
        const animal_vision::DetectionEvent::ConstPtr& message)
    {
        if (!message->target_found)
        {
            detection_latched_ = false;
            latched_event_id_ = message->event_id;
            latched_first_seen_ = ros::Time(0, 0);

            sendYolo(
                false,
                message->class_id,
                message->animal_count,
                0,
                0,
                message->event_id);

            ROS_INFO_THROTTLE(
                2.0,
                "RK_LINK YOLO target_found=0 "
                "class_id=%u count=%u event_id=%u",
                static_cast<unsigned int>(
                    message->class_id),
                static_cast<unsigned int>(
                    message->animal_count),
                static_cast<unsigned int>(
                    message->event_id));
            return;
        }

        if (!detection_latched_ ||
            latched_event_id_ != message->event_id ||
            message->first_seen != latched_first_seen_)
        {
            int16_t matched_x_cm = 0;
            int16_t matched_y_cm = 0;
            double match_delta_ms = 0.0;

            if (!findNearestUwb(
                    message->first_seen,
                    &matched_x_cm,
                    &matched_y_cm,
                    &match_delta_ms))
            {
                sendYolo(
                    false,
                    message->class_id,
                    message->animal_count,
                    0,
                    0,
                    message->event_id);

                ROS_WARN_THROTTLE(
                    1.0,
                    "YOLO event %u has no valid UWB match; "
                    "history=%zu max_delta_ms=%.1f",
                    static_cast<unsigned int>(
                        message->event_id),
                    uwb_history_.size(),
                    vision_uwb_max_delta_ms_);
                return;
            }

            latched_event_id_ = message->event_id;
            latched_first_seen_ = message->first_seen;
            latched_x_cm_ = matched_x_cm;
            latched_y_cm_ = matched_y_cm;
            detection_latched_ = true;

            ROS_INFO(
                "YOLO event %u latched first_seen=%.6f "
                "x_cm=%d y_cm=%d match_delta_ms=%.1f",
                static_cast<unsigned int>(
                    latched_event_id_),
                message->first_seen.toSec(),
                static_cast<int>(latched_x_cm_),
                static_cast<int>(latched_y_cm_),
                match_delta_ms);
        }

        sendYolo(
            true,
            message->class_id,
            message->animal_count,
            latched_x_cm_,
            latched_y_cm_,
            latched_event_id_);

        ROS_INFO_THROTTLE(
            1.0,
            "RK_LINK YOLO target_found=1 "
            "class_id=%u count=%u event_id=%u "
            "x_cm=%d y_cm=%d",
            static_cast<unsigned int>(
                message->class_id),
            static_cast<unsigned int>(
                message->animal_count),
            static_cast<unsigned int>(
                latched_event_id_),
            static_cast<int>(latched_x_cm_),
            static_cast<int>(latched_y_cm_));
    }

void onK230(
    const animal_vision::K230Detection::ConstPtr& message)
{
    const bool online = message->online;
    const bool target_found =
        online && message->target_found;

    const uint8_t class_id =
        online ? message->class_id : 0U;

    const uint8_t animal_count =
        online ? message->animal_count : 0U;

    const uint8_t confidence_percent =
        online ? message->confidence_percent : 0U;

    std::vector<uint8_t> payload;
    payload.reserve(12U);

    rk_link::appendU8(
        payload,
        rk_link::kK230PayloadVersion);

    rk_link::appendU8(
        payload,
        online ? 1U : 0U);

    rk_link::appendU8(
        payload,
        target_found ? 1U : 0U);

    rk_link::appendU8(
        payload,
        class_id);

    rk_link::appendU8(
        payload,
        animal_count);

    rk_link::appendU8(
        payload,
        confidence_percent);

    rk_link::appendU32Le(
        payload,
        message->event_id);

    rk_link::appendU8(
        payload,
        message->sequence);

    rk_link::appendU8(
        payload,
        0U);

    const bool sent =
        writeFrame(
            rk_link::packFrame(
                rk_link::kMsgK230,
                nextSeq(),
                payload));

    if (sent)
    {
        ROS_INFO_THROTTLE(
            1.0,
            "RK_LINK K230 online=%u found=%u "
            "class=%u count=%u confidence=%u "
            "event=%u source_seq=%u",
            online ? 1U : 0U,
            target_found ? 1U : 0U,
            static_cast<unsigned int>(
                class_id),
            static_cast<unsigned int>(
                animal_count),
            static_cast<unsigned int>(
                confidence_percent),
            static_cast<unsigned int>(
                message->event_id),
            static_cast<unsigned int>(
                message->sequence));
    }
}

uint8_t nextSeq()
    {
        return seq_++;
    }

private:

    
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Subscriber uwb_sub_;
    ros::Subscriber vision_sub_;
    ros::Subscriber k230_sub_;
    ros::Timer reconnect_timer_;
    serial::Serial serial_;

    std::string port_name_;
    std::string uwb_topic_;
    std::string vision_event_topic_;
    std::string k230_topic_;
    int baudrate_;
    double send_rate_hz_;
    double uwb_history_sec_;
    double vision_uwb_max_delta_ms_;
    int coordinate_abs_limit_cm_;
    int max_history_samples_;



    uint8_t seq_;
    ros::Time last_uwb_send_;
    std::deque<UwbSample> uwb_history_;

    bool detection_latched_;
    uint32_t latched_event_id_;
    ros::Time latched_first_seen_;
    int16_t latched_x_cm_;
    int16_t latched_y_cm_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "rk_link_sender");

    try
    {
        RKLinkSender sender;
        ros::spin();
    }
    catch (const std::exception& error)
    {
        std::cerr << "RKLinkSender failed: "
                  << error.what() << std::endl;
        ROS_FATAL_STREAM("RKLinkSender failed: " << error.what());
        return 1;
    }
    return 0;
}
