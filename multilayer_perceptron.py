import random
import matplotlib.pyplot as plt
import openpyxl
import math
import numpy as np
import copy


class model():
    learningRate = None
    momentumRate = None
    layer = None
    weight = None
    bias = None
    y = None
    gradient = None
    beforeWeight = None
    beforeBias = None
    sumSquareError = None
    type_ = None

    def __init__(self, list_, type_):
        self.layer = [int(i) for i in list_[0]]
        self.learningRate = float(list_[1])
        self.momentumRate = float(list_[2])
        self.beforeWeight = 0
        self.beforeBias = 0
        self.sumSquareError = 0
        self.type_ = type_

    def reset_WB(self):
        bias = []
        weight = []
        be_bias = []
        be_weight = []
        for i in range(1, len(self.layer)):
            one_layer_b = []
            one_layer_w = []
            one_be_layer_b = []
            one_be_layer_w = []
            for ii in range(self.layer[i]):
                w = []
                be_w = []
                for iii in range(self.layer[i-1]):
                    w.append(round(random.uniform(0.1, 1.0), 2))
                    be_w.append(round(random.uniform(0.1, 1.0), 2))
                one_layer_b.append(round(random.uniform(0.1, 1.0), 2))
                one_be_layer_b.append(round(random.uniform(0.1, 1.0), 2))
                one_layer_w.append(w)
                one_be_layer_w.append(be_w)
            bias.append(np.array(one_layer_b))
            be_bias.append(np.array(one_be_layer_b))
            weight.append(np.array(one_layer_w))
            be_weight.append(np.array(one_be_layer_w))
        self.bias = np.asarray(bias, dtype=object)
        self.weight = np.asarray(weight, dtype=object)

        self.beforeWeight = np.asarray(be_weight, dtype=object)
        self.beforeBias = np.asarray(be_bias, dtype=object)

    def forward(self, x, d_out):
        self.sumSquareError = 0
        x_ = np.array(x)
        for_y = []
        for_y.append(x_)
        for i in range(len(self.weight)):
            result = []
            for ii in range(len(self.weight[i])):
                result.append(x_ @ self.weight[i][ii])
            x_ = self.sigmoid(np.array(result)+self.bias[i])
            for_y.append(x_)
        # เก็บ output ของทุก node ที่ผ่าน sigmoid มาแล้ว
        self.y = np.array(for_y, dtype=object)

        if self.type_ == 'regression':
            part = (for_y[len(for_y)-1]-d_out.tolist())**2
            self.sumSquareError = part/d_out.size
        else:
            part = (for_y[len(for_y)-1]-d_out)**2
            self.sumSquareError = part.sum()/d_out.size

    def train(self, x, d_out):
        self.forward(x, copy.deepcopy(d_out))
        self.backward(np.array(d_out))

    def backward(self, d_out):
        error = copy.deepcopy(d_out) - self.y[len(self.y)-1]  # คำนวณ error

        # คำนวณ gradient output
        gradient_o = error * self.sigmoid(self.y[len(self.y)-1], True)

        lists = []  # เอาไว้เก็บ gradient ของทุก node
        lists.append(gradient_o)
        gradient = gradient_o
        for i in range(len(self.y)-2, 0, -1):  # วนลูปหา gradient ของทุก node
            w = self.weight[i]
            result = self.sigmoid(self.y[i], True) * (gradient @ w)
            lists.append(result)
            gradient = result
        lists = lists[::-1]  # reverse lists ที่เก็บ gradient
        # เปลี่ยน lists เป็น gradient
        gradient = np.asarray(lists, dtype=object)
        newWeight = copy.deepcopy(self.weight)
        y = copy.deepcopy(self.y)
        for i in range(len(newWeight)):
            for ii in range(len(newWeight[i])):
                newWeight[i][ii] = gradient[i][ii]*y[i]

        deltaWeight = (self.momentumRate * (self.weight -
                                            self.beforeWeight)) + (self.learningRate * newWeight)
        deltaWeight = np.array(deltaWeight)

        deltaBias = (self.momentumRate * (self.bias - self.beforeBias)
                     ) + (self.learningRate * gradient)
        deltaBias = np.array(deltaBias)

        self.beforeWeight = copy.deepcopy(self.weight)
        self.weight = self.weight + deltaWeight

        self.beforeBias = copy.deepcopy(self.bias)
        self.bias = self.bias + deltaBias

    def sigmoid(self, i_, diff=False):
        if diff == True:
            return i_ - (i_ ** 2)
        return 1/(1 + np.exp(-1*i_))

    def minmaxNormalization(list_in, new_min, new_max):
        result_ = []
        for list_ in list_in:
            max_ = 0
            min_ = 1000000
            result = []

            for i in list_:
                i = float(i)
                if i > max_:
                    max_ = i
                if i < min_:
                    min_ = i

            for ii in list_:
                ii = float(ii)
                temp = (((ii-min_)/(max_-min_))*(new_max-new_min))+new_min
                result.append(temp)
            result_.append(result)
        return result_


select = None
select = input("select enter 1 (regression) or 2 (classification):\n")
if select == "1":

    f = open("Flood data set.txt", "r")
    count = 0
    M = []
    for i in f:
        if count > 1:
            temp = i[0:len(i)-1].split('\t')
            M.append(temp)
        count += 1
    M = np.array(M)
    Data = minmaxNormalization(M.T, 0.1, 0.9)
    D_out = Data[len(Data)-1:len(Data)]
    D_out = np.array(D_out)
    D_out = D_out[0]
    Data = Data[0:len(Data)-1]
    Data = np.array(Data).T

    list_ = []
    print("Input Size =", len(M[0]) - 1)
    list_.append(
        input("Please enter a numbers of layers: e.g. 2,3,2\n").split(','))
    list_[0].insert(0, str(len(M[0]) - 1))
    list_[0].append("1")
    list_.append(input("Please enter a learning rate:\n"))
    list_.append(input("Please enter a momentum rate:\n"))
    print("layer: ", list_[0])
    print("learning rate: ", list_[1])
    print("momentum rate: ", list_[2])
    print(list_)

    regression = model(list_, 'regression')
    regression.reset_WB()

    sum_ = 0
    for i in range(10):

        part = int(Data.shape[0]/10)
        start = i*part

        print('cross validation : ', i)

        if i == 9:
            end = Data.shape[0]
            test_input = Data[start:end]
            test_output = D_out[start:end]
        else:
            end = start+part
            test_input = Data[start:end]
            test_output = D_out[start:end]

        rang = list(range(start, end))
        train_input = np.delete(Data, rang, axis=0)
        train_output = np.delete(D_out, rang)

        for i_ in range(1, 501):  # train 500 epoch
            Data_ = train_input
            D_out_ = train_output
            while Data_.shape[0] != 0:
                r = random.randrange(0, Data_.shape[0], 1)
                regression.train(Data_[r], D_out_[r])
                D_out_ = np.delete(D_out_, r)
                Data_ = np.delete(Data_, r, axis=0)
        sum__ = 0
        print('shape: ', test_input.shape[0])
        for ii_ in range(test_input.shape[0]):
            regression.forward(test_input[ii_], test_output[ii_])
            sum__ += regression.sumSquareError
        sum_ += sum__/test_input.shape[0]
        print('sumSquareError:', format(regression.sumSquareError[0], '.40g'))
    sum_ = (sum_/10)*100
    print("error : ", format(sum_[0], '.40g'))
    print('......')

elif select == "2":
    confusion_matrix = [[0, 0], [0, 0]]

    f = open("cross.pat", "r")
    data = []
    p = []
    count = 0
    for i in f:
        split_ = i.split('\n')
        split_.pop(len(split_)-1)
        split_ = split_[0].split(' ')
        if count % 3 == 0 and count != 0:
            p = []
        elif count % 3 == 1:
            split_.pop(1)
            p.extend(split_)
        elif count % 3 == 2:
            p.extend(split_)
            data.append(p)
        count += 1
    data = np.array(data, dtype=float)
    Data = minmaxNormalization(data, 0.1, 0.9)
    temp = np.array(Data).T
    D_out = temp[len(temp)-2:len(temp)]
    D_out = np.array(D_out.T)
    Data = temp[0:len(temp)-2]
    Data = np.array(Data).T

    list_ = []
    print("Input Size =", len(Data[0]))
    list_.append(
        input("Please enter a numbers of layers: e.g. 2,3,2\n").split(','))
    list_[0].insert(0, str(len(Data[0])))
    list_[0].append(len(D_out[0]))
    list_.append(input("Please enter a learning rate:\n"))
    list_.append(input("Please enter a momentum rate:\n"))
    print("layer: ", list_[0])
    print("learning rate: ", list_[1])
    print("momentum rate: ", list_[2])
    print(list_)

    classification = model(list_, 'classification')
    classification.reset_WB()

    sum_ = 0
    for i in range(10):
        part = int(Data.shape[0]/10)
        start = i*part
        print('cross validation : ', i)

        if i == 9:
            end = Data.shape[0]
            test_input = Data[start:end]
            test_output = D_out[start:end]
        else:
            end = start+part
            test_input = Data[start:end]
            test_output = D_out[start:end]
        rang = list(range(start, end))
        train_input = np.delete(Data, rang, axis=0)
        train_output = np.delete(D_out, rang, axis=0)

        for i_ in range(1, 501):  # train 500 epoch
            Data_ = copy.deepcopy(train_input)
            D_out_ = copy.deepcopy(train_output)
            while Data_.shape[0] != 0:
                r = random.randrange(0, Data_.shape[0], 1)
                classification.train(Data_[r], D_out_[r])
                D_out_ = np.delete(D_out_, r, axis=0)
                Data_ = np.delete(Data_, r, axis=0)
        sum__ = 0
        for ii_ in range(test_input.shape[0]):
            classification.forward(test_input[ii_], test_output[ii_])
            sum__ += classification.sumSquareError
        sum_ += sum__/test_input.shape[0]
        print('sumSquareError', classification.sumSquareError)
#         print('sumSquareError:',format(classification.sumSquareError[0],'.40g'))
    sum_ = (sum_/10)*100
    print("error:", sum_)
    print('......')

    while Data.shape[0] != 0:
        r = random.randrange(0, Data.shape[0], 1)
        classification.forward(Data[r], D_out[r])
        index = len(classification.y)-1
        if D_out[r][0] > D_out[r][1]:  # D_out = 1 0
            if classification.y[index][0] >= classification.y[index][1]:  # result = 1 0
                confusion_matrix[0][0] += 1
            else:  # result = 0 1
                confusion_matrix[0][1] += 1
        elif D_out[r][0] < D_out[r][1]:  # D_out = 0 1
            if classification.y[index][0] <= classification.y[index][1]:  # result = 1 0
                confusion_matrix[1][0] += 1
            else:  # result = 0 1
                confusion_matrix[1][1] += 1
        D_out = np.delete(D_out, r, axis=0)
        Data = np.delete(Data, r, axis=0)


#     print("error : ",format(sum_,'.40g'))

else:
    print("Try again.")
