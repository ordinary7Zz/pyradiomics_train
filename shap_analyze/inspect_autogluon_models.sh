#!/bin/bash

#########################################################################
# 脚本：inspect_autogluon_models.sh
# 用途：在Ubuntu系统下快速查看AutoGluon模型的leaderboard和ensemble权重
# 
# 用法：
#   ./shap_analyze/inspect_autogluon_models.sh <model_dir>
#   ./shap_analyze/inspect_autogluon_models.sh ./autogluon_model_20260430_120000/
#
# 依赖：
#   - Python 3.8+
#   - pandas, autogluon已安装
#
# 功能：
#   1. 显示leaderboard（模型排名）
#   2. 显示ensemble权重（各模型权重比例）
#   3. 推荐SHAP分析命令
#
#########################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印错误信息
error() {
    echo -e "${RED}❌ 错误: $1${NC}" >&2
    exit 1
}

# 打印警告信息
warning() {
    echo -e "${YELLOW}⚠️  警告: $1${NC}" >&2
}

# 打印成功信息
success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# 打印信息
info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# 打印使用说明
show_usage() {
    cat << EOF
用法：
    ./shap_analyze/inspect_autogluon_models.sh <model_dir> [python_executable]

参数：
    <model_dir>          (必需) AutoGluon模型目录，通常包含predictor.pkl
    [python_executable]  (可选) Python可执行文件路径，默认为'python3'

示例：
    # 使用默认Python（python3）
    ./shap_analyze/inspect_autogluon_models.sh ./autogluon_model/

    # 使用特定的Python环境（conda虚拟环境）
    ./shap_analyze/inspect_autogluon_models.sh ./autogluon_model/ /opt/conda/envs/thyroid/bin/python

    # 使用当前虚拟环境的Python
    source /path/to/venv/bin/activate
    ./shap_analyze/inspect_autogluon_models.sh ./autogluon_model/

输出：
    - Leaderboard：所有训练的模型及其分数排名
    - Ensemble Weights：ensemble中各模型的权重分配
    - 推荐的SHAP分析命令

EOF
}

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 检查参数
if [[ $# -lt 1 ]]; then
    echo -e "${RED}错误：缺少必需参数 <model_dir>${NC}"
    echo ""
    show_usage
    exit 1
fi

MODEL_DIR="$1"
PYTHON_CMD="${2:-python3}"

# 验证Model目录存在
if [[ ! -d "$MODEL_DIR" ]]; then
    error "模型目录不存在: $MODEL_DIR"
fi

# 检查predictor.pkl存在
if [[ ! -f "$MODEL_DIR/predictor.pkl" ]]; then
    warning "在 $MODEL_DIR 中未找到 predictor.pkl，可能不是有效的AutoGluon模型目录"
fi

# 检查Python命令是否可用
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    error "Python命令不可用: $PYTHON_CMD\n请确保Python已正确安装，或指定正确的Python路径"
fi

# 获取Python版本
PYTHON_VERSION=$("$PYTHON_CMD" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
info "使用Python版本: $PYTHON_VERSION ($PYTHON_CMD)"

# 检查必要的包
echo ""
info "检查依赖包..."

check_package() {
    local package=$1
    local import_name=${2:-$1}
    if "$PYTHON_CMD" -c "import $import_name" 2>/dev/null; then
        success "$package 已安装"
    else
        error "$package 未安装。请运行: pip install $package"
    fi
}

check_package "pandas"
check_package "autogluon" "autogluon.tabular"

# 检查inspect脚本存在
INSPECT_SCRIPT="$SCRIPT_DIR/inspect_autogluon_models.py"
if [[ ! -f "$INSPECT_SCRIPT" ]]; then
    error "inspect_autogluon_models.py 不存在于 $SCRIPT_DIR"
fi

success "所有依赖检查完成"

# 运行脚本
echo ""
info "正在运行模型检查..."
echo ""

"$PYTHON_CMD" "$INSPECT_SCRIPT" "$MODEL_DIR"

RESULT=$?

echo ""
if [[ $RESULT -eq 0 ]]; then
    success "模型检查完成！"
    
    # 提供后续建议
    echo ""
    echo -e "${BLUE}=============== 后续步骤 ===============${NC}"
    echo ""
    echo "1. 根据上面显示的模型信息，选择要分析的模型"
    echo ""
    echo "2. 运行SHAP分析（推荐方式）："
    echo -e "   ${YELLOW}python shap_analyze_autogluon_fixed.py \\${NC}"
    echo -e "     ${YELLOW}--model_dir $MODEL_DIR \\${NC}"
    echo -e "     ${YELLOW}--train_csv <训练CSV路径> \\${NC}"
    echo -e "     ${YELLOW}--plot_beeswarm_for <模型名称1> <模型名称2> \\${NC}"
    echo -e "     ${YELLOW}--plot_waterfall${NC}"
    echo ""
    echo "3. 获取更多帮助："
    echo -e "   ${YELLOW}python shap_analyze_autogluon_fixed.py --help${NC}"
    echo ""
    echo -e "${BLUE}========================================${NC}"
else
    error "模型检查失败，请检查错误信息"
fi

exit $RESULT
