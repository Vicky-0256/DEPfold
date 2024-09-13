#!/bin/bash

# 设置每批处理的文件数量
batch_size=1000

# 获取需要处理的文件列表
files=$(git ls-files)

# 计算总文件数
total_files=$(echo "$files" | wc -l)

# 初始化计数器
processed=0

echo "开始处理，总文件数：$total_files"

# 使用 xargs 批量处理文件
echo "$files" | xargs -n $batch_size git rm --cached -r

# 更新进度
processed=$((processed + batch_size))
echo "已处理 $processed / $total_files 个文件"

echo "处理完成"

# 提交更改
git commit -m "Removed cached files in batches"