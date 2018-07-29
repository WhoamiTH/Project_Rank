import glob
import re

def digit(x):
    if str.isdigit(x) or x == '.':
        return True
    else:
        return False

def alpha(x):
    if str.isalpha(x) or x == ' ':
        return True
    else:
        return False

def point(x):
    return x == '.'

def divide_digit(x):
    d = filter(digit, x)
    item = ''
    for i in d:
        item += i
    if len(item) == 0:
        return 0.0
    else:
        p = filter(point, item)
        itemp = ''
        for i in p:
            itemp += i
        # print(itemp)
        if len(itemp) > 1:
            return 0.0
        else:
            return float(item)

def divide_alpha(x):
    a = filter(alpha, x)
    item = ''
    for i in a:
        item += i
    return item

def divide_alpha_digit(x):
    num = divide_digit(x)
    word = divide_alpha(x)
    return word,num

def initlist():
    gp = []
    gr = []
    ga = []
    agtp = []
    agtea = []
    aga = []
    tt = []
    rt = []
    return gp,gr,ga,agtp,agtea,aga,tt,rt

def aver(l):
    return sum(l)/len(l)

def read_file(file_name):
    f = open(file_name,'r')
    gp,gr,ga,agtp,agtea,aga,tt,rt = initlist()
    for i in f:
        word,num = divide_alpha_digit(i)
        # print(word)
        # print(num)
        if word == 'the general precision is ':
            gp.append(num)
        if word == 'the general recall is ':
            gr.append(num)
        if word == 'the general accuracy is ':
            ga.append(num)
        if word == 'the average group top precision is ':
            agtp.append(num)
        if word == 'the average group top exact accuracy is ':
            agtea.append(num)
        if word == 'the average group accuracy is ':
            aga.append(num)
        if word == 'the  time training time is ':
            tt.append(float(str(num)[1:-1]))
        if word == 'the  time running time is ':
            rt.append(float(str(num)[1:-1]))
    av_gp = aver(gp)
    av_gr = aver(gr)
    av_ga = aver(ga)
    av_aptp = aver(agtp)
    av_agtea = aver(agtea)
    av_aga = aver(aga)
    av_tt = aver(tt)
    av_rt = aver(rt)
    return av_gp,av_gr,av_ga,av_aptp,av_agtea,av_aga,av_tt,av_rt

def writerecord(r,l):
    for i in l:
        r.write(f'{i:.4f}\t')
    r.write('\n')

path = '/home/th/WorkSpace/python/python/Project/Result/'
resultdir = 'a_result/'
goal_filename = 'LR_diff_mirror'
goal_dir_add = path + goal_filename + '/*'
result_dir_add = path + resultdir + goal_filename +'.txt'
# print(goal_dir_add)
# print(result_dir_add)
file_list = glob.glob(goal_dir_add)
file_list.sort()
gplist,grlist,galist,agtplist,agtealist,agalist,ttlist,rtlist = initlist()
for file_name in file_list:
    print(file_name)
    av_gp, av_gr, av_ga, av_aptp, av_agtea, av_aga, av_tt, av_rt = read_file(file_name)
    gplist.append(av_gp)
    grlist.append(av_gr)
    galist.append(av_ga)
    agtplist.append(av_aptp)
    agtealist.append(av_agtea)
    agalist.append(av_aga)
    ttlist.append(av_tt)
    rtlist.append(av_rt)

record = open(result_dir_add,'w')
writerecord(record,gplist)
writerecord(record,grlist)
writerecord(record,galist)
writerecord(record,agtplist)
writerecord(record,agtealist)
writerecord(record,agalist)
writerecord(record,ttlist)
writerecord(record,rtlist)