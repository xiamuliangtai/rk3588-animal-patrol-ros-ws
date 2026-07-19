#include <ros/ros.h>
#include <serial/serial.h>

#include <animal_vision/K230Detection.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>

namespace
{

constexpr uint8_t kHead1 = 0xA5;
constexpr uint8_t kHead2 = 0x5A;
constexpr uint8_t kProtocolVersion = 0x01;
constexpr uint8_t kDetectionMessage = 0x01;
constexpr uint8_t kDetectionPayloadLength = 12;
constexpr uint8_t kMaximumPayloadLength = 32;

uint16_t crc16Update(uint16_t crc, uint8_t value)
{
    crc ^= value;

    for (uint8_t bit = 0; bit < 8U; ++bit)
    {
        if ((crc & 0x0001U) != 0U)
        {
            crc = static_cast<uint16_t>(
                (crc >> 1U) ^ 0xA001U);
        }
        else
        {
            crc = static_cast<uint16_t>(crc >> 1U);
        }
    }

    return crc;
}

uint32_t readU32Le(const uint8_t* data)
{
    return static_cast<uint32_t>(data[0]) |
           (static_cast<uint32_t>(data[1]) << 8U) |
           (static_cast<uint32_t>(data[2]) << 16U) |
           (static_cast<uint32_t>(data[3]) << 24U);
}

struct K230Packet
{
    uint8_t sequence = 0U;
    bool target_found = false;
    uint8_t class_id = 0U;
    uint8_t animal_count = 0U;
    uint8_t confidence_percent = 0U;
    uint32_t event_id = 0U;
    uint32_t device_time_ms = 0U;
};

class K230FrameParser
{
public:
    K230FrameParser()
    {
        reset();
    }

    bool feed(uint8_t value, K230Packet* output)
    {
        switch (state_)
        {
        case WAIT_HEAD_1:
            if (value == kHead1)
            {
                state_ = WAIT_HEAD_2;
            }
            break;

        case WAIT_HEAD_2:
            if (value == kHead2)
            {
                state_ = READ_VERSION;
            }
            else if (value != kHead1)
            {
                state_ = WAIT_HEAD_1;
            }
            break;

        case READ_VERSION:
            version_ = value;
            crc_calculated_ = crc16Update(0xFFFFU, value);
            state_ = READ_MESSAGE_TYPE;
            break;

        case READ_MESSAGE_TYPE:
            message_type_ = value;
            crc_calculated_ =
                crc16Update(crc_calculated_, value);
            state_ = READ_SEQUENCE;
            break;

        case READ_SEQUENCE:
            sequence_ = value;
            crc_calculated_ =
                crc16Update(crc_calculated_, value);
            state_ = READ_LENGTH;
            break;

        case READ_LENGTH:
            payload_length_ = value;
            payload_index_ = 0U;
            crc_calculated_ =
                crc16Update(crc_calculated_, value);

            if (payload_length_ > kMaximumPayloadLength)
            {
                ++format_error_count_;
                reset();
            }
            else if (payload_length_ == 0U)
            {
                state_ = READ_CRC_LOW;
            }
            else
            {
                state_ = READ_PAYLOAD;
            }
            break;

        case READ_PAYLOAD:
            payload_[payload_index_++] = value;
            crc_calculated_ =
                crc16Update(crc_calculated_, value);

            if (payload_index_ >= payload_length_)
            {
                state_ = READ_CRC_LOW;
            }
            break;

        case READ_CRC_LOW:
            crc_received_ = value;
            state_ = READ_CRC_HIGH;
            break;

        case READ_CRC_HIGH:
        {
            crc_received_ |=
                static_cast<uint16_t>(value) << 8U;

            const bool crc_ok =
                crc_received_ == crc_calculated_;

            const bool format_ok =
                version_ == kProtocolVersion &&
                message_type_ == kDetectionMessage &&
                payload_length_ == kDetectionPayloadLength;

            if (!crc_ok)
            {
                ++crc_error_count_;
                reset();
                return false;
            }

            if (!format_ok)
            {
                ++format_error_count_;
                reset();
                return false;
            }

            output->sequence = sequence_;
            output->target_found =
                payload_[0] != 0U;
            output->class_id = payload_[1];
            output->animal_count = payload_[2];
            output->confidence_percent = payload_[3];
            output->event_id =
                readU32Le(payload_ + 4);
            output->device_time_ms =
                readU32Le(payload_ + 8);

            ++valid_frame_count_;
            reset();
            return true;
        }
        }

        return false;
    }

    void reset()
    {
        state_ = WAIT_HEAD_1;
        version_ = 0U;
        message_type_ = 0U;
        sequence_ = 0U;
        payload_length_ = 0U;
        payload_index_ = 0U;
        crc_calculated_ = 0xFFFFU;
        crc_received_ = 0U;
    }

    uint32_t validFrameCount() const
    {
        return valid_frame_count_;
    }

    uint32_t crcErrorCount() const
    {
        return crc_error_count_;
    }

    uint32_t formatErrorCount() const
    {
        return format_error_count_;
    }

private:
    enum State : uint8_t
    {
        WAIT_HEAD_1 = 0,
        WAIT_HEAD_2,
        READ_VERSION,
        READ_MESSAGE_TYPE,
        READ_SEQUENCE,
        READ_LENGTH,
        READ_PAYLOAD,
        READ_CRC_LOW,
        READ_CRC_HIGH
    };

    State state_;
    uint8_t version_;
    uint8_t message_type_;
    uint8_t sequence_;
    uint8_t payload_length_;
    uint8_t payload_index_;
    uint8_t payload_[kMaximumPayloadLength];

    uint16_t crc_calculated_;
    uint16_t crc_received_;

    uint32_t valid_frame_count_ = 0U;
    uint32_t crc_error_count_ = 0U;
    uint32_t format_error_count_ = 0U;
};

}  // namespace

class K230SerialReceiver
{
public:
    K230SerialReceiver()
        : nh_(),
          pnh_("~"),
          baudrate_(115200),
          poll_rate_hz_(200.0),
          offline_timeout_sec_(1.0),
          reopen_period_sec_(1.0),
          online_(false),
          last_open_attempt_(0.0),
          last_valid_frame_(0.0)
    {
        pnh_.param<std::string>(
            "port", port_name_, "/dev/k230_vision");
        pnh_.param("baudrate", baudrate_, 115200);
        pnh_.param<std::string>(
            "topic", topic_name_, "/k230/detection");
        pnh_.param(
            "poll_rate", poll_rate_hz_, 200.0);
        pnh_.param(
            "offline_timeout",
            offline_timeout_sec_,
            1.0);
        pnh_.param(
            "reopen_period",
            reopen_period_sec_,
            1.0);

        publisher_ =
            nh_.advertise<animal_vision::K230Detection>(
                topic_name_,
                10,
                true);

        serial_.setPort(port_name_);
        serial_.setBaudrate(
            static_cast<uint32_t>(baudrate_));
        serial::Timeout serial_timeout =
            serial::Timeout::simpleTimeout(20);

        serial_.setTimeout(serial_timeout);

        publishOffline();

        ROS_INFO_STREAM(
            "K230SerialReceiver configured port="
            << port_name_
            << " baudrate=" << baudrate_
            << " topic=" << topic_name_
            << " offline_timeout="
            << offline_timeout_sec_);
    }

    void run()
    {
        ros::WallRate rate(
            std::max(1.0, poll_rate_hz_));

        while (ros::ok())
        {
            ros::spinOnce();

            if (!serial_.isOpen())
            {
                tryOpenSerial();
            }
            else
            {
                pollSerial();
            }

            updateWatchdog();
            rate.sleep();
        }

        closeSerial();
    }

private:
    void tryOpenSerial()
    {
        const ros::WallTime now =
            ros::WallTime::now();

        if (last_open_attempt_.toSec() > 0.0 &&
            (now - last_open_attempt_).toSec() <
                reopen_period_sec_)
        {
            return;
        }

        last_open_attempt_ = now;

        try
        {
            serial_.open();

            if (serial_.isOpen())
            {
                parser_.reset();

                ROS_INFO_STREAM(
                    "K230 serial opened: "
                    << port_name_
                    << " baudrate="
                    << baudrate_);
            }
        }
        catch (const std::exception& error)
        {
            ROS_WARN_THROTTLE(
                2.0,
                "K230 serial open failed: %s",
                error.what());

            closeSerial();
        }
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
            ROS_WARN(
                "K230 serial close failed: %s",
                error.what());
        }

        parser_.reset();
    }

    void pollSerial()
    {
        try
        {
            size_t available = serial_.available();

            if (available == 0U)
            {
                return;
            }

            available =
                std::min<size_t>(available, 512U);

            const std::string bytes =
                serial_.read(available);

            for (const char raw_value : bytes)
            {
                const uint8_t value =
                    static_cast<uint8_t>(
                        static_cast<unsigned char>(
                            raw_value));

                K230Packet packet;

                if (parser_.feed(value, &packet))
                {
                    publishPacket(packet);
                }
            }

            ROS_WARN_THROTTLE(
                2.0,
                "K230 parser stats valid=%u "
                "crc_error=%u format_error=%u",
                static_cast<unsigned int>(
                    parser_.validFrameCount()),
                static_cast<unsigned int>(
                    parser_.crcErrorCount()),
                static_cast<unsigned int>(
                    parser_.formatErrorCount()));
        }
        catch (const std::exception& error)
        {
            ROS_WARN(
                "K230 serial read failed: %s",
                error.what());

            closeSerial();
            setOffline();
        }
    }

    void publishPacket(const K230Packet& packet)
    {
        const bool was_online = online_;

        online_ = true;
        last_valid_frame_ = ros::WallTime::now();

        animal_vision::K230Detection message;
        message.header.stamp = ros::Time::now();
        message.header.frame_id = "k230";
        message.online = true;
        message.target_found =
            packet.target_found;
        message.class_id = packet.class_id;
        message.animal_count =
            packet.animal_count;
        message.confidence_percent =
            packet.confidence_percent;
        message.event_id = packet.event_id;
        message.device_time_ms =
            packet.device_time_ms;
        message.sequence = packet.sequence;

        publisher_.publish(message);

        if (!was_online)
        {
            ROS_INFO("K230 link is online");
        }

        ROS_INFO_THROTTLE(
            1.0,
            "K230 detection online=1 found=%u "
            "class=%u count=%u confidence=%u "
            "event=%u seq=%u",
            packet.target_found ? 1U : 0U,
            static_cast<unsigned int>(
                packet.class_id),
            static_cast<unsigned int>(
                packet.animal_count),
            static_cast<unsigned int>(
                packet.confidence_percent),
            static_cast<unsigned int>(
                packet.event_id),
            static_cast<unsigned int>(
                packet.sequence));
    }

    void updateWatchdog()
    {
        if (!online_)
        {
            return;
        }

        const double age_sec =
            (ros::WallTime::now() -
             last_valid_frame_).toSec();

        if (age_sec > offline_timeout_sec_)
        {
            ROS_WARN(
                "K230 link timeout: no valid "
                "frame for %.3f seconds",
                age_sec);

            setOffline();
        }
    }

    void setOffline()
    {
        if (!online_)
        {
            return;
        }

        online_ = false;
        publishOffline();
    }

    void publishOffline()
    {
        animal_vision::K230Detection message;
        message.header.stamp = ros::Time::now();
        message.header.frame_id = "k230";
        message.online = false;
        message.target_found = false;
        message.class_id = 0U;
        message.animal_count = 0U;
        message.confidence_percent = 0U;
        message.event_id = 0U;
        message.device_time_ms = 0U;
        message.sequence = 0U;

        publisher_.publish(message);
    }

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Publisher publisher_;

    serial::Serial serial_;
    K230FrameParser parser_;

    std::string port_name_;
    std::string topic_name_;

    int baudrate_;
    double poll_rate_hz_;
    double offline_timeout_sec_;
    double reopen_period_sec_;

    bool online_;
    ros::WallTime last_open_attempt_;
    ros::WallTime last_valid_frame_;
};

int main(int argc, char** argv)
{
    ros::init(
        argc,
        argv,
        "k230_serial_receiver");

    try
    {
        K230SerialReceiver receiver;
        receiver.run();
    }
    catch (const std::exception& error)
    {
        ROS_FATAL(
            "K230SerialReceiver failed: %s",
            error.what());

        std::cerr
            << "K230SerialReceiver failed: "
            << error.what()
            << std::endl;

        return 1;
    }

    return 0;
}
