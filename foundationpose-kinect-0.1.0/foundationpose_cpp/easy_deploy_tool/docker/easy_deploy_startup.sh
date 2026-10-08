#!/bin/bash

# 全局配置
FUNCTIONS_DIR="$( cd "$(dirname "$0")" && pwd )"  # 功能脚本存放目录
declare -A loaded_modules  # 已加载模块记录

# 通用功能函数
go_back() { [ ${#menu_stack[@]} -gt 0 ] && current_menu_name="${menu_stack[-1]}" && unset 'menu_stack[-1]'; }

exit_program() { echo "系统已退出"; exit 0; }

# 动态加载模块
load_module() {
    local module=$1
    local module_path="${FUNCTIONS_DIR}/${module}.sh"

    [ -z "${loaded_modules[$module]}" ] && {
        [ -f "$module_path" ] || { echo "错误：模块 $module 不存在"; exit 1; }
        set --; source "$module_path"
        loaded_modules[$module]=1
        echo "已加载模块：$module"
    }
}

# 菜单定义（不再包含返回/退出选项）
declare -A main_menu=(
    ["title"]="主菜单"
    ["prompt"]="请选择设备类型："
    ["options"]="1|nvidia_gpu|menu|nvidia_menu \
                 2|jetson|menu|jetson_menu \
                 3|rknn|menu|rknn_menu"
)

declare -A nvidia_menu=(
    ["title"]="Device Type: nvidia_gpu"
    ["prompt"]="Choose Version of TensorRT and Ubuntu:"
    ["options"]="1|trt8_u2004|func|nvidia_gpu_trt8_u2004 \
                 2|trt8_u2204|func|nvidia_gpu_trt8_u2204 \
                 3|trt10_u2204|func|nvidia_gpu_trt10_u2204"
    ["module"]="build_docker"
)

declare -A jetson_menu=(
    ["title"]="Device Type: jetson"
    ["prompt"]="Choose Version of TensorRT and Ubuntu:"
    ["options"]="1|trt8_u2004|func|jetson_trt8_u2004 \
                 2|trt8_u2204|func|jetson_trt8_u2204 \
                 3|trt10_u2204|func|jetson_trt10_u2204"
    ["module"]="build_docker"
)

declare -A rknn_menu=(
    ["title"]="Device Type: rk3588 or other"
    ["prompt"]="Choose Version of RKNN and Ubuntu:"
    ["options"]="1|rknn_230_u2204|func|rknn_230_u2204"
    ["module"]="build_docker"
)

# 显示菜单系统
show_menu() {
    clear
    local menu_name=$1
    declare -n menu="$menu_name"

    # 预加载模块
    [ -n "${menu[module]}" ] && load_module "${menu[module]}"

    # 显示界面
    echo "========================"
    echo "${menu["title"]}"
    echo "========================"

    IFS=' ' read -ra opts <<< "${menu["options"]}"
    for opt in "${opts[@]}"; do
        IFS='|' read -ra parts <<< "$opt"
        printf "%-2s) %s\n" "${parts[0]}" "${parts[1]}"
    done

    # 导航选项
    [ "$menu_name" != "main_menu" ] && echo "b ) 返回上级"
    echo "q ) 退出系统"
}

# 处理用户选择
handle_choice() {
    local menu_name=$1
    local choice=$2
    declare -n menu="$menu_name"

    case $choice in
        q) exit_program ;;
        b) [ "$menu_name" != "main_menu" ] && go_back ;;
        *)
            IFS=' ' read -ra opts <<< "${menu["options"]}"
            for opt in "${opts[@]}"; do
                IFS='|' read -ra parts <<< "$opt"
                if [[ "${parts[0]}" == "$choice" ]]; then
                    case "${parts[2]}" in
                        menu)
                            menu_stack+=("$menu_name")
                            current_menu_name="${parts[3]}"
                            ;;
                        func)
                            "${parts[3]}"  # 执行目标函数
                            exit 0
                            ;;
                    esac
                    return 0
                fi
            done
            echo "无效选项：$choice"
            ;;
    esac
    return 1
}

# 初始化运行环境
current_menu_name="main_menu"
declare -a menu_stack

# 主程序循环
while true; do
    show_menu "$current_menu_name"
    echo -e "\n${main_menu["prompt"]}"
    read -p "请输入选项: " choice

    if handle_choice "$current_menu_name" "${choice,,}"; then
        continue
    else
        read -p "输入错误，按回车重新选择..."
    fi
done
