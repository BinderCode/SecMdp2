#!/bin/bash
set -e

BLUE='\033[1;34m'
NC='\033[0m'

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" >/dev/null 2>&1 && pwd )"
python_dir="$script_dir/occlum_instance/image/opt/python-occlum"


cd occlum_instance && rm -rf image
copy_bom -f ../pytorch.yaml --root image --include-dir /opt/occlum/etc/template

if [ ! -d $python_dir ];then
    echo "Error: cannot stat '$python_dir' directory"
    exit 1
fi
#//user_space_size LibOS进程可用的enclave内存的总大小   kernel_space_heap_size  LibOS内核的堆大小
#kernel_space_stack_size  LibOS内核的栈大小  max_num_of_threads LibOS线程/进程的最大cd occ数目
new_json="$(jq '.resource_limits.user_space_size = "24000MB" |     

                .resource_limits.kernel_space_heap_size = "1024MB" |
                .resource_limits.max_num_of_threads = 256 |
                .env.default += ["PYTHONHOME=/opt/python-occlum"]' Occlum.json)" && \
echo "${new_json}" > Occlum.json
occlum build

# Test instance for 2 nodes distributed pytorch training
#cp -r ../occlum_instance ../occlum_instance_2

# Run the python demo
#echo -e "${BLUE}occlum run /bin/python3 demo.py${NC}"
occlum run /bin/python3  valp/valp/test_consistency_oram.py
