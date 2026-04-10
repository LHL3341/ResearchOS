#!/bin/bash
cd "$(dirname "$0")"

# 用上级目录名作为 PDF 文件名（项目名/paper 结构，取项目名）
PAPER_NAME="$(basename "$(cd .. && pwd)")"

pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
bibtex main > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1

# 重命名为论文小名
if [ -f main.pdf ]; then
    cp main.pdf "${PAPER_NAME}.pdf"
    echo "✓ ${PAPER_NAME}.pdf"
else
    echo "✗ Compilation failed, check main.log"
fi
