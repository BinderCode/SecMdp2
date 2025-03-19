#!/bin/bash
# 定义函数 generate_ca_files()：这个函数生成证书授权（CA）文件，以及一个用于测试的私钥和证书。这些文件将用于GLOO（一个专门为集群和分布式环境设计的通信库）在运行分布式PyTorch训练时进行TLS加密通信。
# 定义函数 build_instance()：这个函数创建一个新的Occlum实例，然后复制BOM（Bill of Materials）文件中定义的文件和目录到新实例的image目录。之后，它修改Occlum的配置文件（Occlum.json），以增加资源限制、环境变量等。最后，它执行 occlum build 来构建新的Occlum实例。
# 执行以上定义的函数：首先生成CA文件，然后构建Occlum实例。
# 创建一个额外的Occlum实例：将刚刚创建的Occlum实例复制到一个新目录，用于模拟分布式PyTorch训练的多节点环境。
# set -e
#WORLD_SIZE指定将参与训练的进程数   设置为1程序能跑起来
#RANK是每个训练过程的唯一标识符。

#OMP_NUM_THREADS一般可以设置为物理CPU核心数。
BLUE='\033[1;34m'
NC='\033[0m'

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" >/dev/null 2>&1 && pwd )"
python_dir="$script_dir/occlum_instance/image/opt/python-occlum"


function generate_ca_files() 
{
    cn_name=${1:-"localhost"}
    # Generate CA files
    openssl req -x509 -nodes -days 1825 -newkey rsa:2048 -keyout myCA.key -out myCA.pem -subj "/CN=${cn_name}"
    # Prepare test private key
    openssl genrsa -out test.key 2048
    # Use private key to generate a Certificate Sign Request
    openssl req -new -key test.key -out test.csr -subj "/C=CN/ST=Shanghai/L=Shanghai/O=Ant/CN=${cn_name}"
    # Use CA private key and CA file to sign test CSR
    openssl x509 -req -in test.csr -CA myCA.pem -CAkey myCA.key -CAcreateserial -out test.crt -days 825 -sha256
}

function build_instance()
{
    #rm -rf occlum_instance* && occlum new occlum_instance
    pushd occlum_instance
    rm -rf image
    copy_bom -f ../pytorch.yaml --root image --include-dir /opt/occlum/etc/template

    if [ ! -d $python_dir ];then
        echo "Error: cannot stat '$python_dir' directory"
        exit 1
    fi

                #   .env.untrusted += [ "MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "OMP_NUM_THREADS", "HOME" ] |
                #    .env.default += ["GLOO_DEVICE_TRANSPORT=TCP_TLS"] |
                #     .env.default += ["GLOO_DEVICE_TRANSPORT_TCP_TLS_PKEY=/ppml/certs/test.key"] |
                #     .env.default += ["GLOO_DEVICE_TRANSPORT_TCP_TLS_CERT=/ppml/certs/test.crt"] |
                #     .env.default += ["GLOO_DEVICE_TRANSPORT_TCP_TLS_CA_FILE=/ppml/certs/myCA.pem"] |

                #.env.default += [ "MASTER_ADDR=127.0.0.1", "MASTER_PORT=29500" ] 
    new_json="$(jq '.resource_limits.user_space_size = "35000MB" |
                    .resource_limits.kernel_space_heap_size = "1024MB" |
                    .resource_limits.max_num_of_threads = 256 |
                    .env.untrusted += [ "MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "OMP_NUM_THREADS", "HOME" ] |
                    
                    .env.default += ["PYTHONHOME=/opt/python-occlum"]' Occlum.json)" && \
    echo "${new_json}" > Occlum.json
    occlum build
    popd
}

generate_ca_files
build_instance

# Test instance for 2 nodes distributed pytorch training
#cp -r occlum_instance occlum_instance_2

echo -e "${BLUE}occlum run /bin/python3 test_train/test_train/training_clip_ipy.py${NC}"
#occlum run /bin/python3 test_train/test_train/training_clip_ipy.py