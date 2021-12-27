# decision tree prediction wine quality

from __future__ import division
from __future__ import absolute_import
import math
import sys
from io import open
from itertools import imap
TRAINFILECOLUM = 12
TESTFILECOLUM = 11
LABELCLASSNUM = 3

class decision_tree_node(object):
    u"""
        this class is used represent any tree node in a decision tree
        """
    def __init__(self,isLeafNode):
        u"""
            parameters:
            isLeafNode: whether this node is a leaf node
            """
        self.left = None
        self.right = None
        self.isLeafNode = isLeafNode
        self.attr = None
        self.splitVal = None
        self.label = None
    
    def checkIfLeafNode(self):
        u"""
            return true if self is a leaf node
            """
        return self.isLeafNode
    
    def get_splitVal(self):
        u"""
            return current node's splitval
            """
        return self.splitVal
    
    def get_left(self):
        u"""
            return current node's left child
            """
        return self.left
    
    def get_right(self):
        u"""
            return current node's right child
            """
        return self.right
    
    def get_attr(self):
        u"""
            return current node's attr
            """
        return self.attr
    
    def set_splitVal(self,splitVal):
        u"""
            set current node's splitval
            
            parameters:
            splitval: splitval
            """
        self.splitVal = splitVal
    
    def set_left(self,left):
        u"""
            set current node's left child
            """
        self.left = left
    
    def set_right(self,right):
        u"""
            set current node's right child
            """
        self.right = right
    
    
    def set_label(self,label):
        u"""
            set current node's right label
            """
        self.label = label
    
    def set_attr(self,attr):
        u"""
            set current node's attr
            """
        self.attr = attr
    
    def get_label(self):
        u"""
            get current node's label
            """
        return self.label




class data_set(object):
    u"""
        this class is used to store data from data set file
        """
    def __init__(self,x_num):
        u"""
            parameters:
            x_num: the numbers of the features that the data have
            """
        self.size = 0
        self.label=[]
        self.x=[]
        self.y=[]
        self.x_num = x_num
    
    def get_label(self):
        u"""
            return the label of this data set
            """
        return self.label
    
    def get_x_num(self):
        u"""
            get the features num of current dataset
            """
        return self.x_num
    
    def get_size(self):
        u"""
            get the size of the data
            """
        return self.size
    
    
    def get_x(self):
        u"""
            get the features data
            """
        return self.x
    
    def get_y(self):
        u"""
            get the result column
            """
        return self.y
    
    def set_label(self,label):
        u"""
            get the feature and result name as a list
            
            return:
            a list of feature and result names
            """
        self.label=label
    
    def add(self,x,y):
        u"""
            add a new data to the data set
            """
        self.size = self.size + 1
        self.x.append(x)
        if y != None:
            self.y.append(y)

    def equal_x(self):
        u"""
            check if all the x are the same in current dataset
            """
        if len(self.x) > 1:
            i = 0
            while self.x[i] == self.x[i+1] :
                i = i + 1
                if i == (len(self.x) - 1):
                    return True
                    break
            return False
        else:
            return True

    def equal_y(self):
        u"""
            check if all the y are the same in current dataset
            """
        if len(self.y) > 1:
            i = 0
            while self.y[i] == self.y[i+1]:
                i = i + 1
                if i == (len(self.y) - 1):
                    return True
                    break
            return False
        else:
            return True

    def find_attr(self,attr):
        u"""
            get the attr's num repr in current dataset
            """
        for i in xrange(len(self.label)):
            if attr == self.label[i]:
                return i
        return None

    def exist_mode_in_y(self):
        u"""
            get mode of y
            """
        y_num = {}
        for i in xrange(len(self.y)):
            try:
                y_num[self.y[i]] += 1
            except KeyError:
                y_num[self.y[i]] = 1
                                
        max_y = self.y[0]
        max_num = 0
        for (temp_y,temp_num) in y_num.items():
                                        # print temp_y
                                        # print temp_num
            if temp_num>=max_num:
                max_num = temp_num
                max_y = temp_y
                                                    
        return max_y

def load_data(filename,isTrain):
    u"""
        this method is used to load data from data file to
        
        return: data in the form of
        
        parameters:
        filename: the file path and name of the data set
        isTrain: this param should be True if this input file is a train data set,
        and False if a test set file
        """
    try:
        data_file = open(filename)
    except FileNotFoundError:
        return None

    # get the data file by lines
    # get the label
    line = data_file.readline()

    if isTrain:
        x_num = len(line.split()) - (TRAINFILECOLUM - TESTFILECOLUM)
    else:
        x_num = len(line.split())

    data = data_set(x_num)
    data.set_label(line.split())     #first line is the label 

    line = data_file.readline()      #read the next line 
    while line:
        temp_line_spilts = line.split()
        if isTrain:
        # this is for the python3
            temp = list(imap(float,temp_line_spilts[:x_num]))
        # this should be for python2
        #  temp = map(float,temp_line_spilts[:x_num])
            data.add(temp,float(temp_line_spilts[x_num]))
        else:
            # this is for the python3
            temp = list(imap(float,temp_line_spilts))
            # this should be for python2
            # temp = map(float,temp_line_spilts)
            data.add(temp,None)
        line = data_file.readline()
    data_file.close()

    return data

def DTL(data,minleaf):
    u"""
        this method is used to recursively generate the decision tree accroding to the best spilt var calculated
        
        parameters:
        data: the train data set
        minleaf: the minium leaf node number
        """
    if data.get_size() <= minleaf or data.equal_x() or data.equal_y():
        # create a new leaf node
        node = decision_tree_node(True)
        if data.exist_mode_in_y() != None:
            node.set_label(data.exist_mode_in_y())
        # print data.get_y()[0]
        # print data.exist_mode_in_y()
        # print "----"
        else:
            node.set_label("unknown")
        u"""
            if data.exist_mode_in_y():
            # node.set_label(data.get_mode_in_y())
            node.set_label(data.get_y()[0])
            else:
            node.set_label("unknown")
            """
        return node

    (attr,splitval) = chooseSplit(data)
    node = decision_tree_node(False)
    node.set_attr(attr)
    node.set_splitVal(splitval)

    data_under_splitval = data_set(data.get_x_num())
    data_under_splitval.set_label(data.get_label())
    data_upper_splitval = data_set(data.get_x_num())
    data_upper_splitval.set_label(data.get_label())

    x = data.get_x()
    y = data.get_y()
    for i in xrange(len(x)):
        if x[i][data.find_attr(attr)] <= splitval:
            data_under_splitval.add(x[i],y[i])
        else:
            data_upper_splitval.add(x[i],y[i])

    node.set_left(DTL(data_under_splitval,minleaf))
    node.set_right(DTL(data_upper_splitval,minleaf))
    return node



def chooseSplit(data):
    u"""
        find the best spilt point of the input data by calculating the information gain
        
        parameters:
        data: currentdataset
        
        return: a pair of (bestattr,bestsplitval)
        """
    bestgain = 0.0
    bestattr = None
    bestsplitVal = None
    for attr in data.get_label()[:TESTFILECOLUM]:
        sorted_x = [temp[data.find_attr(attr)] for temp in data.get_x()]
        sorted_x.sort()
        for i in xrange(len(sorted_x)-1):
            
            splitVal = 0.5*(sorted_x[i] + sorted_x[i+1])
            gain = information_gain(attr,splitVal,data)
            if gain > bestgain:
                bestattr = attr
                bestsplitVal = splitVal
                bestgain = gain
    return bestattr,bestsplitVal

def information_gain(attr,splitVal,data):
    u"""
        calculate the given information gain of given attr and splitVal
        
        parameters:
        attr: attr
        splitval: splitvar
        data: current train dataset
        
        return: the information gain of attr and splitval
        """
    totelNumOfSamples = data.get_size()
    I_left = 0.0
    I_right = 0.0
    left_list = []
    right_list = []
    for i in xrange(totelNumOfSamples):
        if data.get_x()[i][data.find_attr(attr)] < splitVal:
            left_list.append(i)
        else:
            right_list.append(i)

    left_label_num = {}
    for x in left_list:
        current_y = data.get_y()[x]
        try:
            left_label_num[current_y] += 1
        except KeyError:
            left_label_num[current_y] = 1
    for (label,num) in left_label_num.items():
        p_label = num / len(left_list)
        I_left -= p_label * math.log(p_label,2)
    
    right_label_num = {}
    for x in right_list:
        current_y = data.get_y()[x]
        try:
            right_label_num[current_y] += 1
        except KeyError:
            right_label_num[current_y] = 1
    for (label,num) in right_label_num.items():
        p_label = num / len(right_list)
        I_right -=  p_label * math.log(p_label,2)

    IR = 0.0
    label_num = {}
    for x in xrange(totelNumOfSamples):
        current_y = data.get_y()[x]
        try:
            label_num[current_y] += 1
        except KeyError:
            label_num[current_y] = 1
    for (label,num) in label_num.items():
        p_label = num / totelNumOfSamples
        IR -=  p_label * math.log(p_label,2)
    
    
    result = IR - (len(left_list) / totelNumOfSamples * I_left) - (len(right_list) / totelNumOfSamples * I_right)
    return result


def predict_DTL(n,data):
    u"""
        using the trained decision tree to predict wine quality
        
        parameters:
        n: decision tree root node
        data: test set data
        
        return n.label
        """
    result = []
    for i in xrange(data.get_size()):
        temp_n = n
        while temp_n.checkIfLeafNode() == False:
            if data.get_x()[i][data.find_attr(temp_n.get_attr())] <= temp_n.splitVal:
                temp_n = temp_n.get_left()
            else:
                temp_n = temp_n.get_right()
        result.append(temp_n.get_label())
    
    return result


def main():
    u"""
        entry of the decison tree prediction
        """
    train_set_path = sys.argv[1]
    test_set_path = sys.argv[2]
    minleaf = int(sys.argv[3])
    data = load_data(train_set_path,True)
    test = load_data(test_set_path,False)
    for y in predict_DTL(DTL(data,minleaf),test):
        print int(y)


def test_train_and_test():
    u"""
        this method is used to test DTL and predictDTL
        """
    data = load_data(u"train",True)
    test = load_data(u"test-sample",False)
    # print(predict_DTL(DTL(data,1),test))
    for y in predict_DTL(DTL(data,1),test):
        print int(y)

if __name__ == u"__main__":
    main()

