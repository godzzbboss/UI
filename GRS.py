# -*- coding: utf-8 -*-

"""
__author__ = "BigBrother"

"""
from tkinter import Entry, Label, StringVar, IntVar, Checkbutton, Listbox, NSEW, END, Tk, ttk
from tkinter.messagebox import showerror, showinfo
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat, savemat

# 讲数据进行切分
# with open("./data/k_inv.csv") as f:
#     data = np.loadtxt(f, delimiter=",")
#
# n = data.shape[0]
# data1 = data[:1000,:]
# data2 = data[1000:2000,:]
# data3 = data[2000:3000,:]
# data4 = data[3000:4000,:]
# data5 = data[4000:,:]
# np.savetxt("./data/k_inv1.csv", data1, delimiter=",")
# np.savetxt("./data/k_inv.csv", data2, delimiter=",")
# np.savetxt("./data/k_inv3.csv", data3, delimiter=",")
# np.savetxt("./data/k_inv4.csv", data4, delimiter=",")
# np.savetxt("./data/k_inv5.csv", data5, delimiter=",")
# exit()


# 创建一个字典，保存所有控件
d = {}
win = Tk()
win.title("test")
win["background"] = "white"

# 获取屏幕的宽度跟高度
screen_width = win.winfo_screenwidth()
screen_height = win.winfo_screenheight()
win.geometry("900x500" + "+" + str(100) + "+" + str(100))
win.resizable(0, 0)
win.update()

# ttk主题
ttk_style = ttk.Style()
ttk_style.theme_use("clam")

cb_value = IntVar()  # 复选框的值
entry_gj_var = StringVar()  # 攻角的值
entry_ma_var = StringVar()  # 马赫数的值
entry_speed_var = StringVar()  # 角速度的值
entry_area_var = StringVar()  # 截面积的值
entry_gj1_var = StringVar()  # 批量模式
entry_gj2_var = StringVar()
entry_speed1_var = StringVar()  # 转速的值
entry_step_var = StringVar()  # 步长
entry_variance_var = StringVar()  # 预测方差


def show():
    """
        点击批量，控制显示不同界面

    :return:
    """
    global entry_gj_var, entry_gj1_var, entry_gj2_var, entry_step_var, entry_area_var, entry_speed_var, entry_ma_var, entry_speed1_var, d

    if cb_value.get() == 1:
        d["entry_gj"].destroy()
        d["label_speed1"].destroy()
        d["entry_speed1"].destroy()
        d["button_confirm"].destroy()
        d["label_ma"].destroy()
        d["entry_ma"].destroy()
        d["label_speed"].destroy()
        d["entry_speed"].destroy()
        d["label_area"].destroy()
        d["entry_area"].destroy()
        d["label_variance"].destroy()
        d["entry_variance"].destroy()

        if entry_gj_var.get() != "":
            entry_gj_var.set("")
        if entry_area_var.get() != "":
            entry_area_var.set("")
        if entry_speed_var.get() != "":
            entry_speed_var.set("")
        if entry_ma_var.get() != "":
            entry_ma_var.set("")
        if entry_speed1_var.get() != "":
            entry_speed1_var.set("")
        if entry_variance_var.get() != "":
            entry_variance_var.set("")

        entry_gj1 = Entry(width=8, textvariable=entry_gj1_var)
        label_to = Label(text="to", width=2, bg="white", font=("Courier New", 12, "bold"))
        entry_gj2 = Entry(width=8, textvariable=entry_gj2_var)

        # 攻角步长
        label_step = Label(text="攻角步长:", bg="white", font=("Courier New", 12, "bold"))
        entry_step = Entry(width=8, textvariable=entry_step_var)

        # 马赫数
        label_ma = Label(text="马赫数:", bg="white", font=("Courier New", 12, "bold"))
        entry_ma = Entry(width=8, textvariable=entry_ma_var)

        # 角速度
        label_speed = Label(text="角速度:", bg="white", font=("Courier New", 12, "bold"))
        entry_speed = Entry(width=8, textvariable=entry_speed_var)

        # 截面积
        label_area = Label(text="截面积:", bg="white", font=("Courier New", 12, "bold"))
        entry_area = Entry(width=8, textvariable=entry_area_var)

        button_confirm = ttk.Button(text="确定", command=predict_pl)

        entry_gj1.grid(row=1, column=1, sticky="W")
        label_to.grid(row=1, column=2)
        entry_gj2.grid(row=1, column=3, sticky="W")

        label_step.grid(row=2, column=0, pady=10, ipadx=5, ipady=5, sticky="e")
        entry_step.grid(row=2, column=1, columnspan=3, sticky="W")

        label_ma.grid(row=3, column=0, pady=10, ipadx=5, ipady=5, sticky="e")
        entry_ma.grid(row=3, column=1, columnspan=3, sticky="W")

        label_speed.grid(row=4, column=0, pady=10, ipadx=5, ipady=5, sticky="e")
        entry_speed.grid(row=4, column=1, columnspan=3, sticky="W")

        label_area.grid(row=5, column=0, pady=10, ipadx=5, ipady=5, sticky="e")
        entry_area.grid(row=5, column=1, columnspan=3, sticky="W")

        button_confirm.grid(row=6, rowspan=2, column=0, columnspan=2, padx=10, pady=30, sticky="w")

        d["entry_gj1"] = entry_gj1
        d["entry_gj2"] = entry_gj2
        d["label_to"] = label_to
        d["label_step"] = label_step
        d["entry_step"] = entry_step
        d["label_ma"] = label_ma
        d["entry_ma"] = entry_ma
        d["label_speed"] = label_speed
        d["entry_speed"] = entry_speed
        d["label_area"] = label_area
        d["entry_area"] = entry_area
        d["button_confirm"] = button_confirm



    else:

        d["entry_gj1"].destroy()
        d["entry_gj2"].destroy()
        d["label_to"].destroy()
        d["label_step"].destroy()
        d["entry_step"].destroy()
        d["label_ma"].destroy()
        d["entry_ma"].destroy()
        d["label_speed"].destroy()
        d["entry_speed"].destroy()
        d["label_area"].destroy()
        d["entry_area"].destroy()
        d["button_confirm"].destroy()

        if entry_gj1_var.get() != "":
            entry_gj1_var.set("")
        if entry_gj2_var.get() != "":
            entry_gj2_var.set("")
        if entry_step_var.get() != "":
            entry_step_var.set("")
        if entry_area_var.get() != "":
            entry_area_var.set("")
        if entry_speed_var.get() != "":
            entry_speed_var.set("")
        if entry_ma_var.get() != "":
            entry_ma_var.set("")

        entry_gj = Entry(width=8, textvariable=entry_gj_var)

        # 马赫数
        label_ma = Label(text="马赫数:", bg="white", font=("Courier New", 12, "bold"))
        entry_ma = Entry(width=8, textvariable=entry_ma_var)

        # 角速度
        label_speed = Label(text="角速度:", bg="white", font=("Courier New", 12, "bold"))
        entry_speed = Entry(width=8, textvariable=entry_speed_var)

        # 截面积
        label_area = Label(text="截面积:", bg="white", font=("Courier New", 12, "bold"))
        entry_area = Entry(width=8, textvariable=entry_area_var)

        # 转速
        label_speed1 = Label(text="转速:", bg="white", font=("Courier New", 12, "bold"))
        entry_speed1 = Entry(width=8)

        # 方差
        label_variance = Label(text="预测方差:", bg="white", font=("Courier New", 12, "bold"))
        entry_variance = Entry(width=8, textvariable=entry_variance_var)

        button_confirm = ttk.Button(text="确定", command=predict)

        d["entry_gj"] = entry_gj
        d["label_ma"] = label_ma
        d["entry_ma"] = entry_ma
        d["label_speed"] = label_speed
        d["entry_speed"] = entry_speed
        d["label_area"] = label_area
        d["entry_area"] = entry_area
        d["label_speed1"] = label_speed1
        d["entry_speed1"] = entry_speed1
        d["label_variance"] = label_variance
        d["entry_variance"] = entry_variance
        d["button_confirm"] = button_confirm

        entry_gj.grid(row=1, column=1, columnspan=3, sticky="W")
        label_ma.grid(row=2, column=0, pady=10, ipadx=5, ipady=5, sticky="e")
        entry_ma.grid(row=2, column=1, columnspan=3, sticky="W")
        label_speed.grid(row=3, column=0, pady=10, ipadx=5, ipady=5, sticky="e")
        entry_speed.grid(row=3, column=1, columnspan=3, sticky="W")
        label_area.grid(row=4, column=0, pady=10, ipadx=5, ipady=5, sticky="e")
        entry_area.grid(row=4, column=1, columnspan=3, sticky="W")
        label_speed1.grid(row=5, column=0, pady=10, ipadx=5, ipady=5, sticky="e")
        entry_speed1.grid(row=5, column=1, columnspan=3, sticky="W")
        label_variance.grid(row=6, column=0, pady=10, ipadx=5, ipady=5, sticky="e")
        entry_variance.grid(row=6, column=1, columnspan=3, sticky="W")
        button_confirm.grid(row=7, rowspan=2, column=0, columnspan=2, padx=10, pady=30, sticky="w")


# 画图
plt.ion()
fig = plt.figure()
fig.canvas.set_window_title("Result")
figmanager = plt.get_current_fig_manager().window
figmanager.setGeometry(screen_width / 2 + 100, screen_height / 2 - 400, 500, 500)
plt.subplot(111)

# plt.xlim()

bar1s = []  # 用来保存画的图像
bar2s = []  # 保存置信区间


def show_result():
    """
        根据选择的文件画图
    :return:
    """
    global bar1s, bar2s, listbox
    inds = listbox.curselection()  # 当前选中的数据
    if len(inds) == 0:
        if len(bar1s) == 0:
            showerror("错误", "请选择要显示的数据！")
        else:  # 将上次画的图删掉
            bar1s_ = bar1s.copy()  # 重要
            bar2s_ = bar2s.copy()
            for bar in bar1s_:
                bar1s.remove(bar)  # 将图像从bars中删除
                bar.remove()  # 将图像从画布上删除

            for bar in bar2s_:
                bar2s.remove(bar)  # 将图像从bars中删除
                bar.remove()  # 将图像从画布上删除
            return

    selected_files = [listbox.get(i) for i in inds]  # 当前选中的文件

    # print(selected_files)
    for selected_file in selected_files:
        # print("aaa")
        result = np.loadtxt(result_root + selected_file, delimiter=",")
        # redults[:,0]:攻角  results[:,1]:转速 results[:,2]:方差
        y_min = np.min(result[:, 1]) - np.max(2 * np.sqrt(result[:, 2])) - 10
        y_max = np.max(result[:, 1]) + np.max(2 * np.sqrt(result[:, 2])) + 10
        # y_min = np.min(result[:, 1])
        # y_max = np.max(result[:, 1])
        x_min = np.min(result[:, 0]) - 2
        x_max = np.max(result[:, 0]) + 2
        bar1, = plt.plot(result[:, 0], result[:, 1])
        # print(111111)
        # 画方差
        bar2 = plt.fill_between(result[:, 0], result[:, 1] - 2 * np.sqrt(result[:, 2]),
                                result[:, 1] + 2 * np.sqrt(result[:, 2]), color="b", alpha=0.2)
        # print(222222)
        plt.ylim((y_min, y_max))
        plt.xlim((x_min, x_max))
        # print(333333)
        bar1s.append(bar1)
        bar2s.append(bar2)
        # print(bar2s)


result_root = "./results/"
data_root = "./data/"


def load_result():
    """
        加载结果
    :return:
    """
    import os
    filenames = os.listdir(result_root)
    # print(filenames)
    # 获取当前listbox中的项
    all_item = listbox.get(0, END)
    # listbox.delete(0,END)
    for filename in filenames:
        if filename not in all_item:  # 如果文件不在当前项中则插入
            listbox.insert(END, filename)

            # listbox.update()


def isnum(value):
    """
        判断value是否为合法数字，整数或浮点数
    :param value:
    :return:
    """
    try:
        value_ = float(value)
    except:
        return False
    return True


# 加载K_inv, K(X,X)
# import time
# import pandas as pd

# start = time.clock()
# pd.set_option("precision", 15)
# 加载逆矩阵
# kinv_path = "./data/k_inv.csv"
# showinfo("提示", "程序正在初始化，点击确定继续。")
# # print(111111111)
# with open(kinv_path) as f:
#     k_inv = pd.read_csv(f, header=None, engine="c", float_precision="round_trip")
# # pd.set_option("precision",10)
# showinfo("提示", "初始化完成，点击确定继续。")
# # print(22222222)
# k_inv = np.matrix(k_inv)

# end = time.clock()
# print(end-start)

# 加载训练数据
# train_data_x_path = "./data/data_train_x.csv"
# train_data_y_path = "./data/data_train_y.csv"
#
# with open(train_data_x_path) as f:
#     train_x = pd.read_csv(f, header=None, engine="c", float_precision="round_trip")
# with open(train_data_y_path) as f:
#     train_y = pd.read_csv(f, header=None, engine="c", float_precision="round_trip")
#
# train_x = np.matrix(train_x)
# train_y = np.matrix(train_y)
#
# train_n = train_x.shape[0]  # 训练样本的个数

# 训练数据的均值跟标准差
# from decimal import Decimal

data_mean = np.matrix([2.678233901, 0.983663833, 0.134607395, 1.107395098])
data_std = np.matrix([4.305732638, 0.267103551, 0.047571711, 0.079783933])


# # 最终均值函数
# mu = 0.019752


# def get_k_value(x1, x2):
#     """
#         x1,x2为两个样本的输入，计算这两个样本对应的核函数的值
#     :param x1:
#     :param x2:
#     :return:
#     """
#     l1 = 0.158382815693785
#     sf1 = 0.159979138825701
#     l2 = 0.151679897939583
#     sf2 = 2.002259928266838
#     l3 = 8.743306292059724
#     sf3 = 24.655615708784332
#     l4 = 0.001162083466432
#     sf4 = 1.379389938610182
#     an1 = 59.504226013320988
#     an2 = 2.910808991324680
#     an3 = 1.000485762149749
#     # n, d = x1.shape
#     k = an1**2*(sf1**2*np.exp(-1/2*(x1[0,0]-x2[0,0])**2/l1**2)
#                 +sf2**2*np.exp(-1/2*(x1[0,1]-x2[0,1])**2/l2**2)
#                 +sf3**2*np.exp(-1/2*(x1[0,2]-x2[0,2])**2/l3**2)
#                 +sf4**2*np.exp(-1/2*(x1[0,3]-x2[0,3])**2/l4**2))+\
#         an2**2*(sf1**2*np.exp(-1/2*(x1[0,0]-x2[0,0])**2/l1**2)*sf2**2*np.exp(-1/2*(x1[0,1]-x2[0,1])**2/l2**2)
#                 +sf1**2*np.exp(-1/2*(x1[0,0]-x2[0,0])**2/l1**2)*sf3**2*np.exp(-1/2*(x1[0,2]-x2[0,2])**2/l3**2)
#                 +sf1**2*np.exp(-1/2*(x1[0,0]-x2[0,0])**2/l1**2)*sf4**2*np.exp(-1/2*(x1[0,3]-x2[0,3])**2/l4**2)
#                 +sf2**2*np.exp(-1/2*(x1[0,1]-x2[0,1])**2/l2**2)*sf3**2*np.exp(-1/2*(x1[0,2]-x2[0,2])**2/l3**2)
#                 +sf2**2*np.exp(-1/2*(x1[0,1]-x2[0,1])**2/l2**2)*sf4**2*np.exp(-1/2*(x1[0,3]-x2[0,3])**2/l4**2)
#                 +sf3**2*np.exp(-1/2*(x1[0,2]-x2[0,2])**2/l3**2)*sf4**2*np.exp(-1/2*(x1[0,3]-x2[0,3])**2/l4**2))+\
#         an3**2*(sf1**2*np.exp(-1/2*(x1[0,0]-x2[0,0])**2/l1**2)*sf2**2*np.exp(-1/2*(x1[0,1]-x2[0,1])**2/l2**2)*sf3**2*np.exp(-1/2*(x1[0,2]-x2[0,2])**2/l3**2)
#                 +sf1**2*np.exp(-1/2*(x1[0,0]-x2[0,0])**2/l1**2)*sf2**2*np.exp(-1/2*(x1[0,1]-x2[0,1])**2/l2**2)*sf4**2*np.exp(-1/2*(x1[0,3]-x2[0,3])**2/l4**2)
#                 +sf1**2*np.exp(-1/2*(x1[0,0]-x2[0,0])**2/l1**2)*sf3**2*np.exp(-1/2*(x1[0,2]-x2[0,2])**2/l3**2)*sf4**2*np.exp(-1/2*(x1[0,3]-x2[0,3])**2/l4**2)
#                 +sf2**2*np.exp(-1/2*(x1[0,1]-x2[0,1])**2/l2**2)*sf3**2*np.exp(-1/2*(x1[0,2]-x2[0,2])**2/l3**2)*sf4**2*np.exp(-1/2*(x1[0,3]-x2[0,3])**2/l4**2))
#     return k

def predict_pl():
    """
        批量预测
    :return:
    """
    if isnum(entry_gj1_var.get()):
        aoa1 = float(entry_gj1_var.get())
    else:
        showerror("错误", "请输入合法起始攻角！")
        return

    if isnum(entry_gj2_var.get()):
        aoa2 = float(entry_gj2_var.get())
    else:
        showerror("错误", "请输入合法结束攻角！")
        return

    if float(entry_gj1_var.get()) > float(entry_gj2_var.get()):
        showerror("错误", "请输入合法攻角值！")
        return

    if isnum(entry_step_var.get()) and float(entry_step_var.get()) > 0:
        step = float(entry_step_var.get())
    else:
        showerror("错误", "请输入合法攻角步长！")
        return

    if isnum(entry_ma_var.get()):
        ma = float(entry_ma_var.get())
    else:
        showerror("错误", "请输入合法马赫数！")
        return

    if isnum(entry_speed_var.get()) and float(entry_speed_var.get()) > 0:
        speed = float(entry_speed_var.get())
    else:
        showerror("错误", "请输入合法角速度！")
        return

    if isnum(entry_area_var.get()) and float(entry_area_var.get()) > 0:
        area = float(entry_area_var.get())
    else:
        showerror("错误", "请输入合法截面积！")
        return

    # 生成测试数据
    aoas = np.array(np.linspace(aoa1, aoa2, (aoa2 - aoa1) / step + 1)).reshape(-1, 1)
    aoas_len = len(aoas)
    mas = np.array([ma] * aoas_len).reshape(-1, 1)
    speeds = np.array([speed] * aoas_len).reshape(-1, 1)
    areas = np.array([area] * aoas_len).reshape(-1, 1)

    test_x = np.concatenate((aoas, mas, speeds, areas), axis=1)

    # 标准化
    test_x = (test_x - data_mean) / np.tile(data_std, (aoas_len, 1))

    # 判断某个文件是否存在test_data1中，如果存在则读取，否则创建并读取
    # 文件名使用参数命名
    filename = str(aoa1) + "_" + str(aoa2) + "_" + str(ma) + "_" + str(speed) + "_" + str(area) + ".mat"
    filepath1 = "./test_data1/train_x.mat"
    filepath2 = "./test_data1/train_y.mat"
    # filepath3 = "./test_data1/train_y.mat"
    filepath3 = "./test_data1/" + filename
    if os.path.exists(filepath3):
        import matlab.engine as engine
        eng = engine.start_matlab()
        eng.cd(os.getcwd(), nargout=0)
        # k_2, k_3 = eng.getK(filepath1, filepath2,nargout =2) # k_2:k(x,x')   k_3:k(x',x')
        speed_mean, speed_variance = eng.getMeanAndS2(filepath1, filepath2, filepath3, nargout=2)
    else:
        savemat("./test_data1/" + filename, mdict={"test_x": test_x})
        import matlab.engine as engine
        eng = engine.start_matlab()
        eng.cd(os.getcwd(), nargout=0)
        # k_2, k_3 = eng.getK(filepath1, filepath2, nargout=2)  # k_2:k(x,x')   k_3:k(x',x')
        speed_mean, speed_variance = eng.getMeanAndS2(filepath1, filepath2, filepath3, nargout=2)

    # test_n = test_x.shape[0]  # 测试样本的个数
    # # 计算k_2, K(X', X)
    # k_2 = np.zeros((test_n, train_n))
    # # print(k_2)
    # for i in range(test_n):
    #     x = np.matrix(test_x[i, :])
    #     for j in range(train_n):
    #         z = train_x[j, :]
    #         # print(z)
    #         k_2[i, j] = get_k_value(x,z)

    # k_2 = np.matrix(k_2)
    #
    # # 计算k_3, K(x',x')
    # k_3 = np.zeros((test_n, test_n))
    # for i in range(test_n):
    #     x = np.matrix(test_x[i, :])
    #     for j in range(test_n):
    #         z = np.matrix(test_x[j, :])
    #         k_3[i, j] = get_k_value(x,z)
    # k_2 = np.matrix(np.array(k_2))
    # k_3 = np.matrix(np.array(k_3))
    # np.set_printoptions(precision=11)
    # print(k_2)
    # 计算转速
    # print("k_2:", k_2.shape)
    # print("k_inv:", k_inv.shape)
    # print("train_y:", train_y.shape)
    # speed_mean = k_2.T * k_inv * train_y
    # speed_variance = np.diag(k_3 - k_2.T * k_inv * k_2).reshape(-1, 1)
    # print(speed_variance)
    # print("aoas:", aoas.shape)
    # print("speed_mean:", speed_mean.shape)
    # print("speed_variance:", speed_variance.shape)

    results = np.concatenate((np.matrix(aoas), speed_mean, speed_variance), axis=1)
    filename = str(aoa1) + "-" + str(aoa2) + "-" + str(ma) + "-" + str(speed) + "-" + str(area) + ".csv"
    np.savetxt("./results/" + filename, results, delimiter=",")

    showinfo("提示", "预测成功！")

    # entry_speed1_var.set(round(float(speed_mean), 2))
    # entry_variance_var.set(round(float(speed_variance), 2))


def predict():
    global entry_speed1_var, entry_variance_var

    if isnum(entry_gj_var.get()):
        aoa = float(entry_gj_var.get())
    else:
        showerror("错误", "请输入合法攻角！")
        return

    if isnum(entry_ma_var.get()):
        ma = float(entry_ma_var.get())
    else:
        showerror("错误", "请输入合法马赫数！")
        return
    if isnum(entry_speed_var.get()) and float(entry_speed_var.get()) > 0:
        speed = float(entry_speed_var.get())
    else:
        showerror("错误", "请输入合法角速度！")
        return
    if isnum(entry_area_var.get()) and float(entry_area_var.get()) > 0:
        area = float(entry_area_var.get())
    else:
        showerror("错误", "请输入合法截面积！")
        return

    test_x = np.matrix([aoa, ma, speed, area])

    # 将数据标准化
    test_x = (test_x - data_mean) / data_std
    # 判断某个文件是否存在test_data2中，如果存在则读取，否则创建并读取
    filepath1 = "./test_data2/train_x.mat"
    filepath2 = "./test_data2/train_y.mat"
    # filepath3 = "./test_data1/train_y.mat"
    filepath3 = "./test_data2/test_x.mat"

    savemat(filepath3, mdict={"test_x": test_x})
    import matlab.engine as engine
    eng = engine.start_matlab()
    eng.cd(os.getcwd(), nargout=0)
    # k_2, k_3 = eng.getK(filepath1, filepath2, nargout=2)  # k_2:k(x,x')   k_3:k(x',x')
    speed_mean, speed_variance = eng.getMeanAndS2(filepath1, filepath2, filepath3, nargout=2)

    # test_n = test_x.shape[0]  # 测试样本的个数
    # # 计算k_2, K(X', X)
    # k_2 = np.zeros((test_n, train_n))
    # # print(k_2)
    # for i in range(test_n):
    #     x = test_x[i, :]
    #     # print(x.shape)
    #     for j in range(train_n):
    #         z = train_x[j, :]
    #         # print(z.shape)
    #         # print(z)
    #         k_2[i, j] = get_k_value(x, z)
    #         # print(k_2)
    # k_2 = np.matrix(k_2)
    #
    # # 计算k_3, K(x',x')
    # k_3 = np.zeros((test_n, test_n))
    # for i in range(test_n):
    #     x = test_x[i, :]
    #     for j in range(test_n):
    #         z = test_x[j, :]
    #         k_3[i, j] = get_k_value(x, z)
    # k_3 = np.matrix(k_3)
    # # 计算转速
    # print("k_2:", k_2.shape)
    # print("k_inv:", k_inv.shape)
    # print("train_y:", train_y.shape)
    # # speed_mean = k_2 * k_inv * (train_y - np.tile(mu, (train_y.shape[0], train_y.shape[1])))
    # speed_mean = k_2 * k_inv * train_y
    # speed_variance = np.diag(k_3 - k_2 * k_inv * k_2.T)

    entry_speed1_var.set(round(float(speed_mean), 2))
    entry_variance_var.set(round(float(speed_variance), 2))


def reset():
    global entry_gj_var, entry_gj1_var, entry_gj2_var, entry_step_var, entry_area_var, entry_speed_var, entry_ma_var, entry_speed1_var
    if entry_gj_var.get() != "":
        entry_gj_var.set("")
    if entry_gj1_var.get() != "":
        entry_gj1_var.set("")
    if entry_gj2_var.get() != "":
        entry_gj2_var.set("")
    if entry_step_var.get() != "":
        entry_step_var.set("")
    if entry_area_var.get() != "":
        entry_area_var.set("")
    if entry_speed_var.get() != "":
        entry_speed_var.set("")
    if entry_ma_var.get() != "":
        entry_ma_var.set("")
    if entry_speed1_var.get() != "":
        entry_speed1_var.set("")
    if entry_variance_var.get() != "":
        entry_variance_var.set("")


"""
    第一部分

"""
# 批量
label_pl = Label(text="批量:", bg="white", font=("Courier New", 12, "bold"))
label_pl.grid(row=0, column=0, pady=30, ipadx=5, ipady=5, sticky="e")
d["label_pl"] = label_pl
pl_cb = Checkbutton(win, variable=cb_value, onvalue=1, offvalue=0, command=show)
pl_cb.grid(row=0, column=1, columnspan=3, sticky="W")
d["pl_cb"] = pl_cb
# 攻角
label_gj = Label(text="攻角:", bg="white", font=("Courier New", 12, "bold"))
label_gj.grid(row=1, column=0, pady=10, ipadx=5, ipady=5, sticky="e")
d["label_gj"] = label_gj

entry_gj = Entry(width=8, textvariable=entry_gj_var)
entry_gj.grid(row=1, column=1, columnspan=3, sticky="W")
d["entry_gj"] = entry_gj
# 马赫数
label_ma = Label(text="马赫数:", bg="white", font=("Courier New", 12, "bold"))
label_ma.grid(row=2, column=0, pady=10, ipadx=5, ipady=5, sticky="e")
d["label_ma"] = label_ma
entry_ma = Entry(width=8, textvariable=entry_ma_var)
entry_ma.grid(row=2, column=1, columnspan=3, sticky="W")
d["entry_ma"] = entry_ma
# 角速度
label_speed = Label(text="角速度:", bg="white", font=("Courier New", 12, "bold"))
label_speed.grid(row=3, column=0, pady=10, ipadx=5, ipady=5, sticky="e")
d["label_speed"] = label_speed
entry_speed = Entry(width=8, textvariable=entry_speed_var)
entry_speed.grid(row=3, column=1, columnspan=3, sticky="W")
d["entry_speed"] = entry_speed
# 截面积
label_area = Label(text="截面积:", bg="white", font=("Courier New", 12, "bold"))
label_area.grid(row=4, column=0, pady=10, ipadx=5, ipady=5, sticky="e")
d["label_area"] = label_area
entry_area = Entry(width=8, textvariable=entry_area_var)
entry_area.grid(row=4, column=1, columnspan=3, sticky="W")
d["entry_area"] = entry_area
# 转速
label_speed1 = Label(text="转速:", bg="white", font=("Courier New", 12, "bold"))
label_speed1.grid(row=5, column=0, pady=10, ipadx=5, ipady=5, sticky="e")
d["label_speed1"] = label_speed1
entry_speed1 = Entry(width=8, textvariable=entry_speed1_var)
entry_speed1.grid(row=5, column=1, columnspan=3, sticky="W")
d["entry_speed1"] = entry_speed1
# 预测方差
label_variance = Label(text="预测方差:", bg="white", font=("Courier New", 12, "bold"))
label_variance.grid(row=6, column=0, pady=10, ipadx=5, ipady=5, sticky="e")
d["label_variance"] = label_variance
entry_variance = Entry(width=8, textvariable=entry_variance_var)
entry_variance.grid(row=6, column=1, columnspan=3, sticky="W")
d["entry_variance"] = entry_variance

button_confirm = ttk.Button(text="确定", command=predict)
button_confirm.grid(row=7, rowspan=2, column=0, columnspan=2, padx=10, pady=30, sticky="w")
d["button_confirm"] = button_confirm

# 重置
button_reset = ttk.Button(text="重置", command=reset)
button_reset.grid(row=7, rowspan=2, column=2, columnspan=2, padx=60, pady=30, sticky="e")
d["button_reset"] = button_reset
# 进度条



"""
    第二部分

"""

button_show = ttk.Button(text="SHOW", command=show_result)
# tk.NSEW充满整个网格
button_show.grid(row=2, rowspan=1, column=5, columnspan=2, sticky=NSEW)

button_load = ttk.Button(text="LOAD", command=load_result)
button_load.grid(row=4, rowspan=1, column=5, columnspan=2, sticky=NSEW)

"""
    第三部分

"""

listbox = Listbox(win, width=50, selectmode="multiple")
listbox["background"] = "white"
listbox["borderwidth"] = 2
listbox.grid(row=0, rowspan=8, column=9, columnspan=9, padx=80, sticky=NSEW)
# scrollbar.grid( column=2, row=0, sticky=N+S)
listbox["font"] = ("Courier New", 10, "bold")
win.mainloop()
