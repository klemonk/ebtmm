import streamlit as st
import cv2
import numpy as np
import time

st.set_page_config(page_title="模板匹配", page_icon="random")

st.title("模板匹配，https://github.com/klemonk/ebtmm")

uploaded_file = st.file_uploader("请选择一张图片作为模板图片", type=["jpg", "png", 'jpeg'])
if uploaded_file is not None:
    # 将传入的文件转为Opencv格式
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    #显示图片
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    st.image(img, caption='原图', use_column_width=True)
    #选择匹配方法
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED',#cv2.TM_CCOEFF和cv2.TM_CCOEFF_NORMED表示相关系数匹配；
                'cv2.TM_CCORR','cv2.TM_CCORR_NORMED',#cv2.TM_CCORR和cv2.TM_CCORR_NORMED表示相关性匹配；
                'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']#cv2.TM_SQDIFF和cv2.TM_SQDIFF_NORMED表示平方差匹配
    method = st.selectbox('请选择模板匹配方式', methods)
    #能选择多张图片
    templates = st.file_uploader("请选择匹配图片", type=["jpg", "png", 'jpeg'], accept_multiple_files=True)

    if templates is not None:
        template_images = []
        for template in templates:
            file_bytes = np.asarray(bytearray(template.read()), dtype=np.uint8)
            template_images.append(cv2.imdecode(file_bytes, 1))

        start = time.time()
    #多张图进行匹配
        for template in template_images:
            w, h = template.shape[:-1]
            result = cv2.matchTemplate(img, template, eval(method))
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if method in ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), 3)

        end = time.time()

        st.image(img, caption='匹配结果', use_column_width=True)

        st.write("匹配耗时：", round(end - start, 2), "秒")
