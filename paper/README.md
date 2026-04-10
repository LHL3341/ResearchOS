# Paper Workspace

## 用法

```bash
# 1. 复制模板到新项目
cp -r /mnt/dhwfile/raise/user/linhonglin/paper-workspace-template /path/to/my-paper
cd /path/to/my-paper

# 2. 修改 main.tex 的标题和内容
# 3. 编译
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## 目录结构

```
.
├── main.tex              # 论文主文件 (NeurIPS 2026 格式)
├── neurips_2026.sty      # 官方样式文件
├── refs.bib              # 参考文献
├── figures/              # 图片 (PDF/SVG/PNG)
├── tables/               # 独立表格文件 (可选)
├── sections/             # 拆分的章节文件 (可选)
└── .claude/skills/       # Claude Code 论文写作 skills
    ├── paper-figure/          # 论文图表生成 (matplotlib 模板)
    ├── scientific-visualization/ # 科学可视化 best practice
    ├── scientific-writing/    # IMRAD 写作指导
    ├── research-paper-writing/ # ML 论文段落级写作 + self-review
    ├── paper-strategist/      # 论文规划策略
    └── paper-composer/        # 论文撰写框架
```

## 可用 Skills

在 Claude Code 中使用：

| Skill | 用途 |
|-------|------|
| `/paper-figure` | 生成 publication-ready 图表 (bar/line/heatmap/pipeline) |
| `/research-paper-writing` | 段落级论文写作、reverse outlining、self-review |
| `/paper-strategist` | 论文整体规划和 story 设计 |
| `/paper-composer` | 按章节结构化撰写 |
| `/scientific-visualization` | 科学可视化原则和 colorblind-safe 设计 |
| `/scientific-writing` | IMRAD 结构、citation 规范、revision checklist |

## 编译

需要 texlive。如果集群上没有：

```bash
# 用 apptainer
apptainer exec /path/to/texlive.sif pdflatex main
# 或安装 texlive
# 参考 install-tl-unx.tar.gz
```

### Claude Code 自动编译 Hook

在 `.claude/settings.json` 中添加以下 hook，每次 Edit/Write `.tex` 或 `.bib` 文件后自动编译：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "if echo \"$TOOL_INPUT\" | grep -qE '\\.(tex|bib)'; then cd /your/project/paper && pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1; fi"
          }
        ]
      }
    ]
  }
}
```