import img2pdf
import os

# 图片路径列表
img_paths = ["logo.png"]
output_pdf_name = "framework2.pdf"

for i in img_paths:
# 转换
    with open(f"{i}.pdf", "wb") as f:
        f.write(img2pdf.convert(i))



print(f"转换完成：{output_pdf_name}")