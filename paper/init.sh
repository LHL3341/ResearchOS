#!/bin/bash
# 初始化新论文工作目录
# 用法: bash init.sh /path/to/my-paper "Paper Title"

set -e

TEMPLATE_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:?用法: bash init.sh /path/to/my-paper \"Paper Title\"}"
TITLE="${2:-YOUR PAPER TITLE}"

if [ -d "$TARGET" ]; then
    echo "ERROR: $TARGET 已存在"
    exit 1
fi

cp -r "$TEMPLATE_DIR" "$TARGET"
rm -f "$TARGET/init.sh"  # 不复制初始化脚本本身

# 替换标题
sed -i "s/YOUR PAPER TITLE/${TITLE}/" "$TARGET/main.tex"

echo "✓ 论文工作目录已创建: $TARGET"
echo "  模板: NeurIPS 2026"
echo "  标题: $TITLE"
echo "  Skills: $(ls "$TARGET/.claude/skills/" | wc -l) 个"
echo ""
echo "下一步:"
echo "  cd $TARGET"
echo "  # 编辑 main.tex 开始写作"
