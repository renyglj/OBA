import time
import datetime
import inspect
import torch


def split_layer_dec(absolute_path):

    bwlist = []
    fwlist = []
    def true_get_bkwd_layer_info(cls):
        ori_fwd = cls.forward     # ori_fwd 是class中的 forward


        def generate_function():

            new_input = 'x = Variable(x.data, requires_grad=True)'
            put_to_input_tmp = 'self.input.append(x)\ntorch.cuda.synchronize()\nst_time_fwd = time.perf_counter()'
            put_to_output_tmp = 'torch.cuda.synchronize()\ned_time_fwd = time.perf_counter()\nself.output.append(x)\nfwd_time_lst.append((ed_time_fwd-st_time_fwd)*1000)'

            fwd_code = inspect.getsource(ori_fwd).split('\n')  # 将源码按行分割为字符串格式并存储在列表中
            new_fwd_code = fwd_code.copy()

            # change embeded code to list
            # 将forward函数分按行存储在二维列表，去掉空行与注释
            new_fwd_code = [[line.strip()] for line in new_fwd_code if line.strip()!='' and not line.strip().startswith('#')]
            
            
            for lst_idx in range(1, len(new_fwd_code)):
                line = new_fwd_code[lst_idx][0]
                if line.strip().startswith('return'):
                    break
                else:
                    new_fwd_code[lst_idx].insert(0, put_to_input_tmp)
                    new_fwd_code[lst_idx].insert(0, new_input)
                    new_fwd_code[lst_idx].append(put_to_output_tmp)
            
            new_fwd_code = [line for lst in new_fwd_code for line in lst]
            new_fwd_code = new_fwd_code[1:-1]

            
            # s store the modified function
            s = ""
            for line in new_fwd_code:
                s += line+'\n'
            return s
            # return new_fwd_code

        def bkwd(self, g):
            # print(self.output)

            for i, output in reversed(list(enumerate(self.output))):
                if i == (len(self.output) - 1):
                    # for last node, use g
                    st_time = time.perf_counter()
                    output.backward(g)
                    torch.cuda.synchronize()
                    ed_time = time.perf_counter()
                    # print('Layer_idx:',i+1, 'backward time(ms):',(ed_time-st_time)*1000)
                    if(len(bwlist) < 21):
                        bwlist.append((ed_time-st_time)*1000)
                    else:
                        bwlist[i]=(bwlist[i]+(ed_time-st_time)*1000)/2
                else:
                    st_time = time.perf_counter()
                    output.backward(self.input[i+1].grad.data)
                    torch.cuda.synchronize()
                    ed_time = time.perf_counter()
                    # print('layer_idx:',i+1, '', self.input[i+1].grad.data.sum(),'time(s):',ed_time-st_time)

                    # print('Layer_idx:',i+1, 'backward time(ms):',(ed_time-st_time)*1000)
                    if(len(bwlist) < 21):
                        bwlist.append((ed_time-st_time)*1000)
                    else:
                        bwlist[i]=(bwlist[i]+(ed_time-st_time)*1000)/2
            print("反向：")
            print(bwlist)

        def new_fwd(self, x):
            self.output = []
            self.input = []

            fwd_time_lst = []
            
            func_s = generate_function()
            # for line in func_lst:
            #     exec(line)
            g={'x':x,'self':self,'fwd_time_lst':fwd_time_lst}
            
            head_list = []
            with open(absolute_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if 'import' in line and 'split_layer' not in line:
                        head_list.append(line.strip())
            # print('\n'.join(head_list))
            must_head =  '\nimport time\nimport torch\nimport torch.nn as nn\nfrom torch.autograd import Variable\nimport torch.nn.functional as F\n'

            # func_s = 'import torch\nimport torch.nn as nn\nfrom torch.autograd import Variable\nimport torch.nn.functional as F\n'+func_s)

            func_s = '\n'.join(head_list)+must_head+func_s
            # print(func_s)
            exec(func_s, g)
            
            # print forward time
            print()
            for i,t in enumerate(fwd_time_lst):
                # print('Layer_idx:',i+1, 'forward time(ms):', t)
                if(len(fwlist)<21):
                    fwlist.append(t)
                else:
                    fwlist[i]=(fwlist[i]+t)/2

                # print(type(t))
            # print(self.input[0])
            # print([t.size() for t in self.output])
            print("正向：")
            print(fwlist)
            return self.output[-1]

        cls.forward = new_fwd
        cls.backward = bkwd

        return cls

    return true_get_bkwd_layer_info





