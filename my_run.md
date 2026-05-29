项目名：pyradiomics_train

这个项目的核心目标是：
把甲状腺二维超声图像和对应的分割 mask 转成 radiomics 特征表，再用 AutoGluon 做表格分类训练，最后在测试集上评估，并可选做 SHAP 可解释性分析。

一、项目整体运行主线
1. 准备输入数据
   - 图像目录 image_dir
   - mask 目录 mask_dir
   - 标签文件 label_json

2. 提取 radiomics 特征
   - 入口脚本：extract_radiomics_2d.py
   - 作用：对每张图像，根据对应 mask 提取一行 radiomics 特征
   - 输出：一个 CSV 特征表，每一行对应一张图

3. 训练分类模型
   - 入口脚本：train_autogluon_tabular.py
   - 作用：读取上一步生成的特征 CSV，用 AutoGluon 训练分类器
   - 输出：模型目录 save_dir，以及 leaderboard.csv

4. 测试模型效果
   - 入口脚本：test_autogluon_tabular.py
   - 作用：加载训练好的模型，在一个或多个测试 CSV 上做评估
   - 输出：test_results.csv

5. 做可解释性分析（可选）
   - 入口脚本：shap_analyze_autogluon.py / shap_analyze_autogluon_fixed.py / shap_analyze_autogluon_assets.py
   - 作用：对训练好的模型做 SHAP 分析
   - 输出：特征重要性、SHAP 值、beeswarm 图、waterfall 图等

二、各部分具体流程

1. 特征提取流程
主文件：extract_radiomics_2d.py
配置文件：radiomics_2d.yaml

运行时输入：
- --image_dir：二维图像目录
- --mask_dir：mask 目录
- --label_json：标签 JSON
- --task：选择标签字段，例如 malignancy、tirads
- --output_csv：输出特征 CSV

脚本内部流程：
1) 读取 label_json
2) 遍历每个样本，拿到 filename 和对应任务标签
3) 根据 filename 去 image_dir 找图像
4) 根据 filename 去 mask_dir 找 mask
5) 把图像读成灰度图，把 mask 读成二值区域
6) 检查 image 和 mask 尺寸是否一致
7) 如果 mask 为空，则该样本报错或跳过
8) 调用 PyRadiomics 提取特征
9) 去掉 diagnostics_* 开头的诊断字段
10) 保留真正的 radiomics 特征，并附加：
    - label
    - filename
11) 汇总成 DataFrame
12) 保存为 output_csv

这里的 radiomics 参数由 radiomics_2d.yaml 控制，当前配置特点是：
- force2D: true，说明按二维特征提取
- normalize: false，不做归一化
- resampledPixelSpacing: null，不做重采样
- label: 1，mask 中前景标签固定为 1
- 开启的特征类包括：
  firstorder、glcm、glrlm、glszm、gldm、ngtdm、shape2D

所以，这一步的本质就是：
图像 + mask -> 一行 radiomics 数值特征

2. 训练流程
主文件：train_autogluon_tabular.py

运行时输入：
- --train_csv：训练特征表
- --test_csv：可选，一个或多个测试特征表
- --label：标签列名，默认 label
- --save_dir：模型输出目录
- --presets：AutoGluon 训练配置
- --time_limit：训练时长限制
- --seed：随机种子
- --eval_metric：评估指标

脚本内部流程：
1) 读取 train_csv
2) 做表格清洗：
   - 检查 label 列是否存在
   - 删除非特征列：image_path、mask_path、filename（如果存在）
   - 删除 label == -1 的样本
   - 把 label 转成 int
3) 根据标签类别数判断任务类型：
   - 2 类：binary
   - 多类：multiclass
4) 初始化 TabularPredictor
5) 调用 predictor.fit() 训练模型
6) 如果提供了 --test_csv，就顺便对每个测试集做 evaluate
7) 最后保存 leaderboard.csv

训练输出主要是：
- save_dir：AutoGluon 模型目录
- save_dir/leaderboard.csv：模型排行结果

3. 测试流程
主文件：test_autogluon_tabular.py

运行时输入：
- --model_dir：训练好的模型目录
- --test_csv：一个或多个测试 CSV
- --test_names：可选，对应测试集名字
- --label：标签列名
- --threshold：二分类阈值
- --out_csv：结果输出路径

脚本内部流程：
1) 从 model_dir 加载 TabularPredictor
2) 逐个读取 test_csv
3) 做和训练阶段相同的数据清洗：
   - 删除 image_path、mask_path、filename
   - 删除 label == -1 的样本
   - label 转 int
4) 用 predictor.evaluate() 计算常规指标
5) 用 predictor.predict_proba() 算预测概率
6) 如果是二分类，再额外计算：
   - AUPRC
   - sensitivity
   - specificity
7) 把每个测试集的结果汇总
8) 保存成 out_csv；如果没指定，默认写到 model_dir/test_results.csv

所以，这一步的本质就是：
模型目录 + 测试特征 CSV -> 评估结果表

4. SHAP 分析流程
主文件：
- shap_analyze_autogluon.py
- shap_analyze_autogluon_fixed.py
- shap_analyze_autogluon_assets.py

作用：
对已经训练好的 AutoGluon 模型做解释，分析每个 radiomics 特征对预测的贡献。

大致流程：
1) 加载训练好的 predictor
2) 读取训练 CSV
3) 选背景样本和解释样本
4) 对主要模型计算 SHAP 值
5) 输出：
   - shap_values.csv
   - feature_importance.csv
   - summary 文本
   - beeswarm 图
   - waterfall 图

这部分不是主训练链路必须步骤，而是训练完成后的附加分析。

三、项目目录里的两种运行模式
scripts 目录分成两类：

1. scripts/gtmask/
   - 使用真实标注 mask 来提取 radiomics 特征
   - 对应 dataset1、dataset2、dataset3、dataset4 的提取、训练、测试、SHAP 脚本

2. scripts/prediction/
   - 使用预测得到的 mask 来提取 radiomics 特征
   - 同样提供 dataset_1 到 dataset_4 的提取、训练、测试、SHAP 脚本

这说明项目实际支持两条实验线：
- GT mask 实验线
- Prediction mask 实验线

两条线的整体步骤一样，只是 mask 来源不同。

四、最标准的实际运行顺序
一个完整实验通常按这个顺序执行：

第 1 步：提取训练集 radiomics 特征
- 运行 extract_radiomics_2d.py
- 得到训练集特征 CSV

第 2 步：提取外部测试集 radiomics 特征
- 继续运行 extract_radiomics_2d.py
- 分别得到各测试集 CSV，例如 TN3K、ThyroidXL、TN5K

第 3 步：训练 AutoGluon 模型
- 运行 train_autogluon_tabular.py
- 输入训练集 CSV
- 输出模型目录

第 4 步：测试模型
- 运行 test_autogluon_tabular.py
- 输入模型目录和若干测试 CSV
- 输出 test_results.csv

第 5 步：做 SHAP 分析（可选）
- 运行 shap_analyze_autogluon_fixed.py 或 shap_analyze_autogluon_assets.py
- 输出解释结果和图表
- 如果要把已经生成的 SHAP 图和超声样本图拼成一张论文风格总图，可以再运行 nature_shap_draft.py
  - 它本身不计算 SHAP，只负责把外部准备好的图片按版式合成
  - 通过 --config 传入一个 JSON 配置文件，里面要显式写出每张图片的路径
  - 配置里通常包含 figure、layout、beeswarm_panels、sample_panels 这几部分
  - 运行后会生成 PNG / PDF 拼图文件，例如 out/nature_shap_draft.png 和 out/nature_shap_draft.pdf

五、项目中最关键的文件
1. extract_radiomics_2d.py
   - 负责从图像和 mask 提取 radiomics 特征

2. radiomics_2d.yaml
   - 负责定义 radiomics 提取参数

3. train_autogluon_tabular.py
   - 负责训练表格分类模型

4. test_autogluon_tabular.py
   - 负责评估模型

5. shap_analyze_autogluon_fixed.py
   - 负责生成较完整的 SHAP 分析结果和图

6. scripts/gtmask/*.sh 与 scripts/prediction/*.sh
   - 这些 shell 脚本相当于整套实验命令模板
   - 直接展示了项目实际怎么串起来运行

六、用一句话概括项目运行逻辑
这个项目先把“图像区域”转成“radiomics 表格特征”，再把“特征表”交给 AutoGluon 训练分类器，最后在多个测试集上评估，并用 SHAP 解释模型。
