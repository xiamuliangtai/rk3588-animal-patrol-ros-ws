#include <ros/ros.h>
#include <serial/serial.h>
#include <std_msgs/String.h>
#include <Uwb_Location/uwb.h>
#include <iostream>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace rk_link
{
constexpr uint8_t kFrameHead1 = 0xAA;
constexpr uint8_t kFrameHead2 = 0x55;
constexpr uint8_t kVersion = 0x01;
constexpr uint8_t kMsgUwb = 0x01;
constexpr uint8_t kMsgYolo = 0x02;
constexpr size_t kMaxPayload = 32;

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

void appendI16Le(std::vector<uint8_t>& out, int16_t value)
{
    appendU16Le(out, static_cast<uint16_t>(value));
}

void appendI32Le(std::vector<uint8_t>& out, int32_t value)
{
    const uint32_t raw = static_cast<uint32_t>(value);
    out.push_back(static_cast<uint8_t>(raw & 0x000000FFU));
    out.push_back(static_cast<uint8_t>((raw >> 8U) & 0x000000FFU));
    out.push_back(static_cast<uint8_t>((raw >> 16U) & 0x000000FFU));
    out.push_back(static_cast<uint8_t>((raw >> 24U) & 0x000000FFU));
}

std::vector<uint8_t> packFrame(uint8_t msg_id, uint8_t seq, const std::vector<uint8_t>& payload)
{
    if (payload.size() > kMaxPayload)
    {
        throw std::runtime_error("RK_LINK payload too long");
    }

    std::vector<uint8_t> body;
    body.reserve(4 + payload.size());
    appendU8(body, kVersion);
    appendU8(body, msg_id);
    appendU8(body, seq);
    appendU8(body, static_cast<uint8_t>(payload.size()));
    body.insert(body.end(), payload.begin(), payload.end());

    const uint16_t crc = crc16Modbus(body.data(), body.size());

    std::vector<uint8_t> frame;
    frame.reserve(2 + body.size() + 2);
    appendU8(frame, kFrameHead1);
    appendU8(frame, kFrameHead2);
    frame.insert(frame.end(), body.begin(), body.end());
    appendU16Le(frame, crc);
    return frame;
}

int32_t metersToCentimeters(float value_m)
{
    return static_cast<int32_t>(std::lround(static_cast<double>(value_m) * 100.0));
}

std::string extractJsonString(const std::string& json, const std::string& key, const std::string& fallback)
{
    const std::string pattern = "\"" + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos)
    {
        return fallback;
    }

    pos = json.find(':', pos + pattern.size());
    if (pos == std::string::npos)
    {
        return fallback;
    }

    pos = json.find('"', pos + 1);
    if (pos == std::string::npos)
    {
        return fallback;
    }

    const size_t end = json.find('"', pos + 1);
    if (end == std::string::npos || end <= pos + 1)
    {
        return fallback;
    }

    return json.substr(pos + 1, end - pos - 1);
}

int extractJsonInt(const std::string& json, const std::string& key, int fallback)
{
    const std::string pattern = "\"" + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos)
    {
        return fallback;
    }

    pos = json.find(':', pos + pattern.size());
    if (pos == std::string::npos)
    {
        return fallback;
    }

    ++pos;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t'))
    {
        ++pos;
    }

    char* end_ptr = nullptr;
    const long value = std::strtol(json.c_str() + pos, &end_ptr, 10);
    if (end_ptr == json.c_str() + pos)
    {
        return fallback;
    }

    return static_cast<int>(value);
}

uint8_t animalToClassId(const std::string& animal)
{
    if (animal == "peacock") return 0;
    if (animal == "wolf") return 1;
    if (animal == "monkey") return 2;
    if (animal == "elephant") return 3;
    if (animal == "tiger") return 4;
    return 0;
}

}  // namespace rk_link

class RKLinkSender
{
public:
    RKLinkSender()
        : nh_(), pnh_("~"), seq_(0), send_rate_hz_(20.0)
    {
        pnh_.param<std::string>("port", port_name_, "/dev/ttyUSB1");
        pnh_.param("baudrate", baudrate_, 500000);
        pnh_.param<std::string>("uwb_topic", uwb_topic_, "/uwb/data");
        pnh_.param<std::string>("vision_topic", vision_topic_, "/vision/result_stable");
        pnh_.param("send_rate", send_rate_hz_, 20.0);

        serial::Timeout timeout = serial::Timeout::simpleTimeout(20);
        serial_.setPort(port_name_);
        serial_.setBaudrate(static_cast<uint32_t>(baudrate_));
        serial_.setTimeout(timeout);
        serial_.open();

        if (!serial_.isOpen())
        {
            throw std::runtime_error("RK_LINK serial open failed: " + port_name_);
        }

        uwb_sub_ = nh_.subscribe(uwb_topic_, 1, &RKLinkSender::onUwb, this);
        vision_sub_ = nh_.subscribe(vision_topic_, 1, &RKLinkSender::onVision, this);

        ROS_INFO_STREAM("RKLinkSender opened " << port_name_
                        << " baudrate=" << baudrate_
                        << " uwb_topic=" << uwb_topic_
                        << " vision_topic=" << vision_topic_
                        << " send_rate=" << send_rate_hz_);
    }

private:
    // void writeFrame(const std::vector<uint8_t>& frame)
    // {
    //     const size_t written = serial_.write(frame);
    //     if (written != frame.size())
    //     {
    //         ROS_WARN_THROTTLE(1.0, "RK_LINK short write: %zu/%zu", written, frame.size());
    //     }
    // }
    void writeFrame(const std::vector<uint8_t>& frame)
    {
        try
        {
            if (!serial_.isOpen())
            {
                ROS_ERROR_THROTTLE(1.0, "RK_LINK serial is not open");
                return;
            }

            const size_t written = serial_.write(frame);
            if (written != frame.size())
            {
                ROS_WARN_THROTTLE(1.0, "RK_LINK short write: %zu/%zu", written, frame.size());
            }
        }
        catch (const std::exception& e)
        {
            ROS_ERROR_THROTTLE(1.0, "RK_LINK serial write failed: %s", e.what());
        }
    }

    void onUwb(const Uwb_Location::uwb::ConstPtr& msg)
    {
        if (send_rate_hz_ > 0.0 && !last_uwb_send_.isZero())
        {
            const ros::Duration min_period(1.0 / send_rate_hz_);
            if ((ros::Time::now() - last_uwb_send_) < min_period)
            {
                return;
            }
        }
        last_uwb_send_ = ros::Time::now();

        const int32_t x_cm = rk_link::metersToCentimeters(msg->x);
        const int32_t y_cm = rk_link::metersToCentimeters(msg->y);
        const int16_t vx_cmps = 0;
        const int16_t vy_cmps = 0;
        const uint16_t quality = 100;
        const uint16_t age_ms = 0;

        std::vector<uint8_t> payload;
        payload.reserve(16);
        rk_link::appendI32Le(payload, x_cm);
        rk_link::appendI32Le(payload, y_cm);
        rk_link::appendI16Le(payload, vx_cmps);
        rk_link::appendI16Le(payload, vy_cmps);
        rk_link::appendU16Le(payload, quality);
        rk_link::appendU16Le(payload, age_ms);

        writeFrame(rk_link::packFrame(rk_link::kMsgUwb, nextSeq(), payload));

        ROS_INFO_THROTTLE(1.0, "RK_LINK UWB sent x_cm=%d y_cm=%d", x_cm, y_cm);
    }

    void onVision(const std_msgs::String::ConstPtr& msg)
    {
        const std::string animal = rk_link::extractJsonString(msg->data, "animal", "none");
        const int count = rk_link::extractJsonInt(msg->data, "count", 0);
        const bool found = (animal != "none" && count > 0);

        const uint8_t target_found = found ? 1 : 0;
        const uint8_t class_id = found ? rk_link::animalToClassId(animal) : 0;
        const uint8_t confidence_percent = found ? 100 : 0;

        std::vector<uint8_t> payload;
        payload.reserve(12);
        rk_link::appendU8(payload, target_found);
        rk_link::appendU8(payload, class_id);
        rk_link::appendU8(payload, confidence_percent);
        rk_link::appendU8(payload, 0);  // reserved
        rk_link::appendI16Le(payload, 0);  // cx_px, unavailable in stable JSON now
        rk_link::appendI16Le(payload, 0);  // cy_px
        rk_link::appendI16Le(payload, 0);  // box_w_px
        rk_link::appendI16Le(payload, 0);  // box_h_px

        writeFrame(rk_link::packFrame(rk_link::kMsgYolo, nextSeq(), payload));

        ROS_INFO_THROTTLE(1.0, "RK_LINK YOLO sent found=%u class_id=%u animal=%s count=%d",
                          target_found, class_id, animal.c_str(), count);
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
    serial::Serial serial_;
    std::string port_name_;
    std::string uwb_topic_;
    std::string vision_topic_;
    int baudrate_;
    double send_rate_hz_;
    uint8_t seq_;
    ros::Time last_uwb_send_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "rk_link_sender");

    try
    {
        RKLinkSender sender;
        ros::spin();
    }
    // catch (const std::exception& e)
    // {
    //     ROS_FATAL_STREAM("RKLinkSender failed: " << e.what());
    //     return 1;
    // }
    catch (const std::exception& e)
    {
        std::cerr << "RKLinkSender failed: " << e.what() << std::endl;
        ROS_FATAL_STREAM("RKLinkSender failed: " << e.what());
        return 1;
    }

    return 0;
}
