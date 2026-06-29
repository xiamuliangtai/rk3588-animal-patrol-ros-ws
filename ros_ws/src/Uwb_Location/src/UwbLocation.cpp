#include <ros/ros.h>
#include <serial/serial.h>
#include <iostream>
#include <string>
#include <string.h>
#include <algorithm>
#include <cmath>
#include "std_msgs/String.h"              //ros定义的String数据类型
#include "Uwb_Location/trilateration.h"
#include "Uwb_Location/uwb.h"
#include <sensor_msgs/Imu.h>

using namespace std;
unsigned char receive_buf[1024] = {0};
vec3d report;
float uwb_range;
float uwb_aoa;
float uwb_range_a1;
float uwb_aoa_a1;
int tagId = 0;
float pdoa_x = -1;
float pdoa_y = -1;
Quaternion q;
int result = 0; 
float velocityac[3],angleac[3];

Quaternion Q;
// Default orientation (identity). If the IMU stream doesn't provide quaternion, we keep identity.
// Q.q0=w, Q.q1=x, Q.q2=y, Q.q3=z

#define MAX_DATA_NUM	1024	//传消息内容最大长度
#define DataHead        'm'       
#define DataHead2        'M'
#define DataTail        '\n'   
unsigned char BufDataFromCtrl[MAX_DATA_NUM];
int BufCtrlPosit_w = 0, BufCtrlPosit_r = 0;
int DataRecord=0, rcvsign = 0;
int range[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
bool uwb_new_position = false;

static const int UWB_ANCHOR_COUNT_FOR_XY = 4;
static const double UWB_TAG_HEIGHT_M = 1.20;  // H problem flight height. Adjust after calibration.

static void reset_ranges()
{
    std::fill(range, range + 8, -1);
}

static void load_anchor_positions(vec3d anchorArray[8])
{
    for (int i = 0; i < 8; ++i)
    {
        anchorArray[i].x = 0.0;
        anchorArray[i].y = 0.0;
        anchorArray[i].z = 0.0;
    }

    // Unit: meter. Keep these values consistent with the real anchor layout.
    anchorArray[0].x = 0.0;
    anchorArray[0].y = 0.0;
    anchorArray[0].z = 2.0;

    anchorArray[1].x = 0.0;
    anchorArray[1].y = 1.8;
    anchorArray[1].z = 2.0;

    anchorArray[2].x = 3.6;
    anchorArray[2].y = 0.0;
    anchorArray[2].z = 2.0;

    anchorArray[3].x = 3.6;
    anchorArray[3].y = 1.8;
    anchorArray[3].z = 2.0;
}

static bool horizontal_range_m(const vec3d& anchor, int range_mm, double* horizontal_range)
{
    if (range_mm <= 0)
    {
        return false;
    }

    const double slant_range = static_cast<double>(range_mm) / 1000.0;
    const double dz = UWB_TAG_HEIGHT_M - anchor.z;
    double horizontal_sq = slant_range * slant_range - dz * dz;

    if (horizontal_sq < -0.05)
    {
        return false;
    }
    if (horizontal_sq < 0.0)
    {
        horizontal_sq = 0.0;
    }

    *horizontal_range = std::sqrt(horizontal_sq);
    return true;
}

static int solve_xy_location(vec3d* solution, const vec3d anchorArray[8], const int distanceArray[8])
{
    bool valid[8] = {false};
    double horizontal_ranges[8] = {0.0};
    int valid_count = 0;
    int ref = -1;

    for (int i = 0; i < UWB_ANCHOR_COUNT_FOR_XY; ++i)
    {
        if (horizontal_range_m(anchorArray[i], distanceArray[i], &horizontal_ranges[i]))
        {
            valid[i] = true;
            ++valid_count;
            if (ref < 0)
            {
                ref = i;
            }
        }
    }

    if (valid_count < 3 || ref < 0)
    {
        return -1;
    }

    const double x0 = anchorArray[ref].x;
    const double y0 = anchorArray[ref].y;
    const double r0 = horizontal_ranges[ref];

    double ata00 = 0.0;
    double ata01 = 0.0;
    double ata11 = 0.0;
    double atb0 = 0.0;
    double atb1 = 0.0;
    int equation_count = 0;

    for (int i = 0; i < UWB_ANCHOR_COUNT_FOR_XY; ++i)
    {
        if (!valid[i] || i == ref)
        {
            continue;
        }

        const double xi = anchorArray[i].x;
        const double yi = anchorArray[i].y;
        const double ri = horizontal_ranges[i];
        const double a = 2.0 * (xi - x0);
        const double b = 2.0 * (yi - y0);
        const double c = r0 * r0 - ri * ri + xi * xi - x0 * x0 + yi * yi - y0 * y0;

        ata00 += a * a;
        ata01 += a * b;
        ata11 += b * b;
        atb0 += a * c;
        atb1 += b * c;
        ++equation_count;
    }

    if (equation_count < 2)
    {
        return -1;
    }

    const double det = ata00 * ata11 - ata01 * ata01;
    if (std::fabs(det) < 1e-9)
    {
        return -1;
    }

    solution->x = (atb0 * ata11 - ata01 * atb1) / det;
    solution->y = (ata00 * atb1 - ata01 * atb0) / det;
    solution->z = 0.0;  // Z is intentionally not calculated from UWB.
    return 2;
}

void receive_deal_func()
{
    vec3d anchorArray[8];
    uwb_new_position = false;
    reset_ranges();
    
    if((receive_buf[0] == 'm') && (receive_buf[1] == 'c'))
    {
        int aid = 0, tid = 0, lnum = 0, seq = 0, mask = 0;
        int rangetime = 0;
        char role = 0;

        // Prefer parsing by field count instead of fixed string length.
        // 8-anchor format
        int n = sscanf((char*)receive_buf,
                       "mc %x %x %x %x %x %x %x %x %x %x %x %x %c%d:%d",
                       &mask,
                       &range[0], &range[1], &range[2], &range[3],
                       &range[4], &range[5], &range[6], &range[7],
                       &lnum, &seq, &rangetime, &role, &tid, &aid);
        if (n == 15)
        {
            printf("mask=0x%02x\nrange[0]=%d(mm)\nrange[1]=%d(mm)\nrange[2]=%d(mm)\nrange[3]=%d(mm)\nrange[4]=%d(mm)\nrange[5]=%d(mm)\nrange[6]=%d(mm)\nrange[7]=%d(mm)\r\n",
                   mask, range[0], range[1], range[2], range[3],
                   range[4], range[5], range[6], range[7]);
            tagId = tid;
        }
        else
        {
            // 4-anchor format
            n = sscanf((char*)receive_buf,
                       "mc %x %x %x %x %x %x %x %x %c%d:%d",
                       &mask,
                       &range[0], &range[1], &range[2], &range[3],
                       &lnum, &seq, &rangetime, &role, &tid, &aid);
            if (n == 11)
            {
                printf("mask=0x%02x\nrange[0]=%d(mm)\nrange[1]=%d(mm)\nrange[2]=%d(mm)\nrange[3]=%d(mm)\r\n",
                       mask, range[0], range[1], range[2], range[3]);
                tagId = tid;
            }
            else
            {
                return;
            }
        }
    }
    //MP0034,0,302,109,287,23,134.2,23.4,23,56
    else if((receive_buf[0] == 'M') && (receive_buf[1] == 'P'))
    {
        char *ptr, *retptr;
        ptr = (char*)receive_buf;
        char cut_data[30][12];
        int cut_count = 0;

        while((retptr = strtok(ptr,",")) != NULL )
        {
            //printf("%s\n", retptr);
            strcpy(cut_data[cut_count], retptr);
            ptr = NULL;
            cut_count++;
            if(cut_count >= 29)
                break;
        }
        
        int tag_id = atoi(cut_data[1]);
        printf("tag_id = %d\n", tag_id);

        float x = (float)atoi(cut_data[2]) / 100.0f;
        printf("x = %.2fm\n", x);
        
        float y = (float)atoi(cut_data[3]) / 100.0f;
        printf("y = %.2fm\n", y);   
        
        float aoa = atof(cut_data[7]);
        printf("aoa = %.2f°\n", aoa);
        
        float dis = (float)atoi(cut_data[4]) / 100.0f;
        printf("dis = %.2fm\n", dis);

        float dis_a1 = -1.0f;
        if (cut_count > 10) dis_a1 = atof(cut_data[10]);
        printf("dis_a1 = %.2f\n", dis_a1);

        float aoa_a1 = 0.0f;
        if (cut_count > 12) aoa_a1 = atof(cut_data[12]);
        printf("aoa_a1 = %.2f°\n", aoa_a1);

        tagId = tag_id;
        uwb_range = dis;
        uwb_range_a1 = dis_a1;
        uwb_aoa = aoa;
        uwb_aoa_a1 = aoa_a1;
        pdoa_x = x;
        pdoa_y = y;
        report.x = x;
        report.y = y;
        report.z = 0.0;
        result = 2;
        uwb_new_position = true;
        return;

    }
    else if((receive_buf[0] == 'm') && (receive_buf[1] == 'i'))
    {
        float rangetime2;
        float acc[3], gyro[3], mag[3];
        float pitch, roll, yaw;

        //mi,981.937,0.63,NULL,NULL,NULL,-2.777783,1.655664,9.075048,-0.004788,-0.014364,-0.001596,T0     //13
        //mi,3.710,0.55,NULL,NULL,NULL,NULL,NULL,NULL,NULL,-1.327881,0.653174,9.577490,-0.004788,-0.013300,-0.002128,T0    //17
        char *ptr, *retptr;
        ptr = (char*)receive_buf;
        char cut_data[30][12];
        int cut_count = 0;

        while((retptr = strtok(ptr,",")) != NULL )
        {
            //printf("%s\n", retptr);
            strcpy(cut_data[cut_count], retptr);
            ptr = NULL;
            cut_count++;
            if(cut_count >= 29)
                break;
        }

        rangetime2 = atof(cut_data[1]);
        
        if(cut_count == 13)  //4anchors
        {
            for(int i = 0; i < 4; i++)
            {
                if(strcmp(cut_data[i+2], "NULL"))
                {
                    range[i] = atof(cut_data[i+2]) * 1000;
                }
                else
                {
                    range[i] = -1;
                }
            }

            for(int i = 0; i < 3; i++)
            {
                acc[i] = atof(cut_data[i+6]);
            }

            for(int i = 0; i < 3; i++)
            {
                gyro[i] = atof(cut_data[i+9]);
            }
            
            velocityac[0] = acc[0];
            velocityac[1] = acc[1];
            velocityac[2] = acc[2];

            angleac[0] = gyro[0];
            angleac[1] = gyro[1];
            angleac[2] = gyro[2];
            printf("4anchors 6axis\n");
            printf("rangetime = %.3f\n", rangetime2);
            printf("range[0] = %d\n", range[0]);
            printf("range[1] = %d\n", range[1]);
            printf("range[2] = %d\n", range[2]);
            printf("range[3] = %d\n", range[3]);
            printf("acc[0] = %.3f\n", acc[0]);
            printf("acc[1] = %.3f\n", acc[1]);
            printf("acc[2] = %.3f\n", acc[2]);
            printf("gyro[0] = %.3f\n", gyro[0]);
            printf("gyro[1] = %.3f\n", gyro[1]);
            printf("gyro[2] = %.3f\n", gyro[2]);
            int tid = atoi((char*)cut_data[12]+1);
            tagId = tid;
            printf("tag_id = %d\n", tagId);

        }
        else if(cut_count == 17)  //8anchors
        {
            for(int i = 0; i < 8; i++)
            {
                if(strcmp(cut_data[i+2], "NULL"))
                {
                    range[i] = atof(cut_data[i+2]) * 1000;
                }
                else
                {
                    range[i] = -1;
                }
            }

            for(int i = 0; i < 3; i++)
            {
                acc[i] = atof(cut_data[i+6+4]);
            }

            for(int i = 0; i < 3; i++)
            {
                gyro[i] = atof(cut_data[i+9+4]);
            }
            velocityac[0] = acc[0];
            velocityac[1] = acc[1];
            velocityac[2] = acc[2];

            angleac[0] = gyro[0];
            angleac[1] = gyro[1];
            angleac[2] = gyro[2];
            printf("8anchors 6axis\n");
            printf("rangetime = %.3f\n", rangetime2);
            printf("range[0] = %d\n", range[0]);
            printf("range[1] = %d\n", range[1]);
            printf("range[2] = %d\n", range[2]);
            printf("range[3] = %d\n", range[3]);
            printf("range[4] = %d\n", range[4]);
            printf("range[5] = %d\n", range[5]);
            printf("range[6] = %d\n", range[6]);
            printf("range[7] = %d\n", range[7]);
            printf("acc[0] = %.3f\n", acc[0]);
            printf("acc[1] = %.3f\n", acc[1]);
            printf("acc[2] = %.3f\n", acc[2]);
            printf("gyro[0] = %.3f\n", gyro[0]);
            printf("gyro[1] = %.3f\n", gyro[1]);
            printf("gyro[2] = %.3f\n", gyro[2]);
            int tid = atoi((char*)cut_data[16]+1);
            tagId = tid;
            printf("tag_id = %d\n", tagId);
        }
        else if(cut_count == 23)  //8anchors 9axis
        {
            for(int i = 0; i < 8; i++)
            {
                if(strcmp(cut_data[i+2], "NULL"))
                {
                    range[i] = atof(cut_data[i+2]) * 1000;
                }
                else
                {
                    range[i] = -1;
                }
            }

            for(int i = 0; i < 3; i++)
            {
                acc[i] = atof(cut_data[i+6+4]);
            }

            for(int i = 0; i < 3; i++)
            {
                gyro[i] = atof(cut_data[i+9+4]);
            }

            for(int i = 0; i < 3; i++)
            {
                mag[i] = atof(cut_data[i+9+3+4]);
            }
            velocityac[0] = acc[0];
            velocityac[1] = acc[1];
            velocityac[2] = acc[2];

            angleac[0] = gyro[0];
            angleac[1] = gyro[1];
            angleac[2] = gyro[2];
            printf("8anchors 9axis\n");
            printf("rangetime = %.3f\n", rangetime2);
            printf("range[0] = %d\n", range[0]);
            printf("range[1] = %d\n", range[1]);
            printf("range[2] = %d\n", range[2]);
            printf("range[3] = %d\n", range[3]);
            printf("range[4] = %d\n", range[4]);
            printf("range[5] = %d\n", range[5]);
            printf("range[6] = %d\n", range[6]);
            printf("range[7] = %d\n", range[7]);
            printf("acc[0] = %.3f\n", acc[0]);
            printf("acc[1] = %.3f\n", acc[1]);
            printf("acc[2] = %.3f\n", acc[2]);
            printf("gyro[0] = %.3f\n", gyro[0]);
            printf("gyro[1] = %.3f\n", gyro[1]);
            printf("gyro[2] = %.3f\n", gyro[2]);
            printf("mag[0] = %.3f\n", mag[0]);
            printf("mag[1] = %.3f\n", mag[1]);
            printf("mag[2] = %.3f\n", mag[2]);
            int tid = atoi((char*)cut_data[22]+1);
            tagId = tid;
            printf("tag_id = %d\n", tagId);
        }
        else
        {
            printf("cut_count = %d\r\n", cut_count);
            return;
        }
    }
    else
    {
        puts("no range message");
        return;
    }

    load_anchor_positions(anchorArray);
    result = solve_xy_location(&report, anchorArray, range);
    uwb_new_position = (result > 0);

    printf("result = %d\n",result);
    printf("x = %f\n",report.x);
    printf("y = %f\n",report.y);
    printf("z unused = %f\n",report.z);

}


void CtrlSerDataDeal()
{
    unsigned char middata = 0;
    static unsigned char dataTmp[MAX_DATA_NUM] = {0};

    while(BufCtrlPosit_r != BufCtrlPosit_w)
    {
        middata = BufDataFromCtrl[BufCtrlPosit_r];
        BufCtrlPosit_r = (BufCtrlPosit_r==MAX_DATA_NUM-1)? 0 : (BufCtrlPosit_r+1);

        if(((middata == DataHead)||(middata == DataHead2))&&(rcvsign == 0))//收到头
        {
            rcvsign = 1;//开始了一个数据帧
            dataTmp[DataRecord++] = middata;//数据帧接收中
        }
        else if((middata != DataTail)&&(rcvsign == 1))
        {
            dataTmp[DataRecord++] = middata;//数据帧接收中
        }
        else if((middata == DataTail)&&(rcvsign == 1))//收到尾
        {
            if(DataRecord != 1)
            {
                rcvsign = 0;
                dataTmp[DataRecord++] = middata;
                dataTmp[DataRecord] = '\0';

                strncpy((char*)receive_buf, (char*)dataTmp, DataRecord);
                printf("receive_buf = %slen = %d\n", receive_buf, DataRecord);
                receive_deal_func(); /*调用处理函数*/
                bzero(receive_buf, sizeof(receive_buf));

                DataRecord = 0;
            }
        }
    }
}

// Simulation helpers and simulation mode
static void simulate_frame(const std::string& frame)
{
    bzero(receive_buf, sizeof(receive_buf));
    std::string f = frame;
    if (f.empty() || f.back() != '\n') f.push_back('\n');
    size_t n = std::min(f.size(), sizeof(receive_buf) - 1);
    memcpy(receive_buf, f.data(), n);
    receive_buf[n] = '\0';
    receive_deal_func();
}

static void run_simulation_step(int step)
{
    // Cycle through different message types to validate parsers.
    switch (step % 5)
    {
    case 0:
        // mc 4-anchor (hex ranges in mm), role + tid:aid at end
        simulate_frame("mc ff 3e8 3f2 44c 4b0 01 02 1234 T0:0");
        break;
    case 1:
        // mc 8-anchor
        simulate_frame("mc ff 3e8 3f2 44c 4b0 514 578 5dc 640 01 02 2345 T0:0");
        break;
    case 2:
        // MP (PDOA) – at least 13 comma-separated fields
        simulate_frame("MP0034,0,302,109,287,23,134.2,23.4,0,0,1.23,0,45.6");
        break;
    case 3:
        // mi 4anchors 6axis (13 fields)
        simulate_frame("mi,1.000,1.1,2.2,3.3,4.4,0.01,0.02,9.80,0.10,0.20,0.30,T0");
        break;
    default:
        // mi 8anchors 6axis (17 fields)
        simulate_frame("mi,2.000,1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,0.01,0.02,9.80,0.10,0.20,0.30,T1");
        break;
    }

    // Ensure a valid quaternion is always published.
    Q.q0 = 1.0f;
    Q.q1 = 0.0f;
    Q.q2 = 0.0f;
    Q.q3 = 0.0f;
}


int main(int argc, char** argv)
{
    setlocale(LC_ALL,"");
	std_msgs::String msg;
	std_msgs::String  msg_mc;
	int  data_size;
	int n;
	int cnt = 0;
    ros::init(argc, argv, "uwb_imu_node");//发布imu,uwb节点
    //创建句柄（虽然后面没用到这个句柄，但如果不创建，运行时进程会出错）
    ros::NodeHandle nh;
    ros::NodeHandle nh1;
    ros::NodeHandle pnh("~");
    bool simulate = false;
    int sim_rate_hz = 20;
    pnh.param("simulate", simulate, false);
    pnh.param("sim_rate", sim_rate_hz, 20);
    ros::Publisher uwb_publisher = nh.advertise<Uwb_Location::uwb>("/uwb/data", 1000);//发布uwb数据  话题名 队列大小
    ros::Publisher IMU_read_pub = nh.advertise<sensor_msgs::Imu>("imu/data", 1000);//发布imu话题

    //创建一个serial类
    serial::Serial sp;

    if (!simulate)
    {
        //创建timeout
        serial::Timeout to = serial::Timeout::simpleTimeout(11);
        //设置要打开的串口名称
        sp.setPort("/dev/ttyUSB7");
        //设置串口通信的波特率
        sp.setBaudrate(115200);
        //串口设置timeout
        sp.setTimeout(to);

        try
        {
            //打开串口
            sp.open();
        }
        catch(serial::IOException& e)
        {
            ROS_ERROR_STREAM("Unable to open port.");
            return -1;
        }

        //判断串口是否打开成功
        if(sp.isOpen())
        {
            ROS_INFO_STREAM("Serial port is opened: /dev/ttyCH341USB0");
        }
        else
        {
            return -1;
        }
    }

    // Default quaternion (identity)
    Q.q0 = 1.0f;
    Q.q1 = 0.0f;
    Q.q2 = 0.0f;
    Q.q3 = 0.0f;

    //发布uwb话题
    Uwb_Location::uwb uwb_data;
    //打包IMU数据
    sensor_msgs::Imu imu_data;

    ros::Rate loop_rate(simulate ? sim_rate_hz : 50);
    int sim_step = 0;

    while(ros::ok())
    {
        bool updated = false;
        uwb_new_position = false;

        if (simulate)
        {
            run_simulation_step(sim_step++);
            updated = true;
        }
        else
        {
            //获取缓冲区内的字节数
            size_t len = sp.available();
            if(len > 0)
            {
                unsigned char usart_buf[1024]={0};
                sp.read(usart_buf, len);

                unsigned char *pbuf;
                unsigned char buf[2014] = {0};

                pbuf = (unsigned char *)usart_buf;
                memcpy(&buf[0], pbuf, len);

                int reallength = (int)len;
                if(reallength != 0)
                {
                    for(int i=0; i < reallength; i++)
                    {
                        BufDataFromCtrl[BufCtrlPosit_w] = buf[i];
                        BufCtrlPosit_w = (BufCtrlPosit_w==(MAX_DATA_NUM-1))? 0 : (1 + BufCtrlPosit_w);
                    }
                }
                CtrlSerDataDeal();
                updated = true;
            }
        }

        if (updated && uwb_new_position)
        {
            //---------------------------------UWB----------------------------------------------------
            uwb_data.header.stamp = ros::Time::now();
            uwb_data.x = report.x;
            uwb_data.y = report.y;
            uwb_data.z = report.z;
            uwb_data.tag_id = tagId;

            uwb_data.d0 = range[0];
            uwb_data.d1 = range[1];
            uwb_data.d2 = range[2];
            uwb_data.d3 = range[3];
            uwb_data.d4 = range[4];
            uwb_data.d5 = range[5];
            uwb_data.d6 = range[6];
            uwb_data.d7 = range[7];

            // PDOA
            uwb_data.aoa = uwb_aoa;
            uwb_data.aoa_a1 = uwb_aoa; // placeholder (will be overwritten below if available)
            uwb_data.aoa_a1 = uwb_aoa_a1;
            uwb_data.distance = uwb_range;
            uwb_data.distance_a1 = uwb_range_a1;
            uwb_data.pdoa_x = pdoa_x;
            uwb_data.pdoa_y = pdoa_y;

            //--------------------------------------IMU------------------------------------------------
            imu_data.header.stamp = uwb_data.header.stamp;
            imu_data.header.frame_id = "base_link";
            imu_data.linear_acceleration.x = velocityac[0];
            imu_data.linear_acceleration.y = velocityac[1];
            imu_data.linear_acceleration.z = velocityac[2];

            imu_data.angular_velocity.x = angleac[0];
            imu_data.angular_velocity.y = angleac[1];
            imu_data.angular_velocity.z = angleac[2];

            imu_data.orientation.x = Q.q1;
            imu_data.orientation.y = Q.q2;
            imu_data.orientation.z = Q.q3;
            imu_data.orientation.w = Q.q0;

            //--------------------------------------话题发布------------------------------------
            uwb_publisher.publish(uwb_data);
            IMU_read_pub.publish(imu_data);
        }

        ros::spinOnce();
        loop_rate.sleep();
    }
    //关闭串口
    if (!simulate)
    {
        sp.close();
    }
    return 0;
}
