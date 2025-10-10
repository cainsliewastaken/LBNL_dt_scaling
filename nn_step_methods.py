import numpy as np
import torch
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
# from nn_jacobian_loss import *

class PEC_step(nn.Module):
    def __init__(self, network, device, time_step = 1): 
        super(PEC_step, self).__init__()
        self.network = network
        self.device = device
        # self.time_step = nn.Parameter(torch.ones(1))
        self.time_step = time_step

    def forward(self, input_batch):
      output_1 = self.network(input_batch.to(self.device)) + input_batch.to(self.device)
      return input_batch.to(self.device) + self.time_step*0.5*(self.network(input_batch.to(self.device))+self.network(output_1))
    

class Euler_step(nn.Module):
    def __init__(self, network, device, time_step = 1): 
        super(Euler_step, self).__init__()
        self.network = network
        self.device = device
        # self.time_step = nn.Parameter(torch.ones(1))
        self.time_step = time_step

    def forward(self, input_batch):
      return input_batch.to(self.device) + self.time_step*(self.network(input_batch.to(self.device)))


class PEC4_step(nn.Module):
    def __init__(self, network, device, time_step = 1): 
        super(PEC4_step, self).__init__()
        self.network = network
        self.device = device
        # self.time_step = nn.Parameter(torch.ones(1))
        self.time_step = time_step

    def forward(self, input_batch):
        temp_out_1 = self.network(input_batch.cuda())
        output_1 = self.time_step*temp_out_1 + input_batch.cuda()
        output_2 = input_batch.cuda() + self.time_step*0.5*(temp_out_1+self.network(output_1))
        output_3 = input_batch.cuda() + self.time_step*0.5*(temp_out_1+self.network(output_2))
        return input_batch.cuda() + self.time_step*0.5*(temp_out_1+self.network(output_3))


class RK4_step(nn.Module):
    def __init__(self, network, device, time_step = 1): 
        super().__init__()
        self.network = network
        self.device = device
        # self.time_step = nn.Parameter(torch.ones(1))
        self.time_step = time_step


    def forward(self, input_batch):
        input_batch = input_batch.to(self.device)
        output_1 = self.network(input_batch)
        output_2 = self.network(input_batch+0.5*output_1)
        output_3 = self.network(input_batch+0.5*output_2)
        output_4 = self.network(input_batch+output_3)

        return input_batch + self.time_step*(output_1+2*output_2+2*output_3+output_4)/6

class Direct_step(nn.Module):
    def __init__(self, network, device, time_step = 1): 
        super(Direct_step, self).__init__()
        self.network = network
        self.device = device

    def forward(self, input_batch):
      return self.network(input_batch.to(self.device))

class Implicit_Euler_step(nn.Module):
    def __init__(self, network, device, num_iters, time_step = 1): 
        super(Implicit_Euler_step, self).__init__()
        self.network = network
        self.device = device
        self.num_iters = num_iters
        # self.time_step = nn.Parameter(torch.ones(1))
        self.time_step = time_step

    def implicit_euler(self, input_batch, output):
        return input_batch.to(self.device) + self.time_step*(self.network(output.to(self.device)))
    
    def change_num_iters(self, num_iters):
        self.num_iters = num_iters
        print('New iters: ',self.num_iters)
        return
    
    def forward(self, input_batch):
        output = input_batch.to(self.device) + self.time_step*(self.network(input_batch.to(self.device)))
        iter = 0
        while iter < self.num_iters:
           output = self.implicit_euler(input_batch, output)
           iter += 1
        return output


class Switch_Euler_step(nn.Module):
    def __init__(self, network, device, num_iters, time_step = 1): 
        super(Switch_Euler_step, self).__init__()
        self.network = network
        self.device = device
        self.num_iters = num_iters
        # self.time_step = nn.Parameter(torch.ones(1))
        self.time_step = time_step

    def change_num_iters(self, num_iters):
        self.num_iters = num_iters
        print('New iters: ',self.num_iters)
        return
    
    def explicit_forward(self, u_0):
       return u_0 + self.network(u_0)
    
    def implicit_forward(self, u_0):
        u_1 = u_0.to(self.device) + self.time_step*(self.network(u_0.to(self.device)))
        iter = 0
        while iter < self.num_iters:
           u_1 = self.implicit_forwards_inner(u_0, u_1)
           iter += 1
        return u_1
    
    def explicit_backwards(self, u_1):
       return u_1 - self.network(u_1)
    
    def implicit_backwards(self, u_1):
        u_0 = u_1.to(self.device) - self.time_step*(self.network(u_1.to(self.device)))
        iter = 0
        while iter < self.num_iters:
           u_0 = self.implicit_backwards_inner(u_1, u_0)
           iter += 1
        return u_0

    def implicit_forwards_inner(self, u_0, u_1):
        return u_0.to(self.device) + self.time_step*(self.network(u_1.to(self.device)))
    
    def implicit_backwards_inner(self, u_0, u_1):
        return u_0.to(self.device) - self.time_step*(self.network(u_1.to(self.device)))
    
    def train_explicit(self, loss_func, u_0, u_1):
       return loss_func(self.explicit_backwards(u_1), u_0)

    def train_implicit(self, loss_func, u_0, u_1):
       return loss_func(self.implicit_backwards(u_1), u_0)
    
    def train_explicit_jacobian(self, u_0, u_1):
       return jacobian_loss(self.explicit_backwards, u_1, u_0)
    
    def train_implicit_jacobian(self, u_0, u_1):
       return jacobian_loss(self.implicit_backwards, u_1, u_0)
    
    def train_explicit_spectral_jacobian(self, u_0, u_1, alpha, beta):
       return spectral_jacobian_loss(self.explicit_backwards, u_1, u_0, alpha, beta)
    
    def train_implicit_spectral_jacobian(self, u_0, u_1, alpha, beta):
       return spectral_jacobian_loss(self.implicit_backwards, u_1, u_0, alpha, beta)
    
    def train_explicit_trace(self, u_0, u_1, alpha, beta):
       return trace_jacobian_loss(self.explicit_backwards, u_1, u_0, alpha, beta)




class Second_order_multistep(nn.Module):
    def __init__(self, network, device, time_step = 1): 
        super(Second_order_multistep, self).__init__()
        self.network = network
        self.device = device
        self.time_step = nn.Parameter(torch.ones(1), requires_grad=False)
        with torch.no_grad():
           self.time_step.data.copy_(time_step)

    def forward(self, input_batch):
      return 2*input_batch[:,1].to(self.device) - input_batch[:,0].to(self.device) + self.time_step.to(self.device)*(self.network(input_batch[:,1].to(self.device)))



class Second_order_multistep_high_order_dt(nn.Module):
    def __init__(self, network, device, time_step = 1): 
        super(Second_order_multistep_high_order_dt, self).__init__()
        self.network = network
        self.device = device
        self.time_step = nn.Parameter(torch.ones(1), requires_grad=False)
        with torch.no_grad():
           self.time_step.data.copy_(time_step)

    def forward(self, input_batch):
      du = (3/2*input_batch[:,-1].to(self.device) - 2*input_batch[:,1].to(self.device) + 1/2*input_batch[:,0])
      return input_batch[:,-1].to(self.device) + du + self.time_step.to(self.device)*(self.network(input_batch[:,-1].to(self.device)))




class Second_order_multistep_high_order_dt_input(nn.Module):
    def __init__(self, network, device, time_step = 1): 
        super(Second_order_multistep_high_order_dt_input, self).__init__()
        self.network = network
        self.device = device
        self.time_step = nn.Parameter(torch.ones(1), requires_grad=True)
        with torch.no_grad():
           self.time_step.data.copy_(time_step)

    def forward(self, input_batch):
      du = (3/2*input_batch[:,2].to(self.device) - 2*input_batch[:,1].to(self.device) + 1/2*input_batch[:,0].to(self.device))
      return input_batch[:,-1].to(self.device) + du + self.time_step.to(self.device)*(self.network(torch.cat([input_batch[:,-1].to(self.device), du], dim=2)))

class Second_order_multistep_3rd_order_udot(nn.Module):
    def __init__(self, network, device, time_step = 1): 
        super(Second_order_multistep_3rd_order_udot, self).__init__()
        self.network = network
        self.device = device
        self.time_step = nn.Parameter(torch.ones(1, dtype=torch.double), requires_grad=True)
        with torch.no_grad():
           self.time_step.data.copy_(time_step)

    def forward(self, input_batch):
      du = 11/6*input_batch[:, 3].to(self.device) - 3*input_batch[:, 2].to(self.device) + 3/2*input_batch[:, 1].to(self.device) - 1/3*input_batch[:, 0].to(self.device)
      net_out = self.time_step.to(self.device)*(self.network(torch.cat([input_batch[:,-1].to(self.device), du], dim=2).float())).double()
      return input_batch[:,-1].to(self.device) + du + net_out.float()



class Third_order_multistep_3rd_order_udot(nn.Module):
    def __init__(self, network, device, time_step = 1): 
        super(Third_order_multistep_3rd_order_udot, self).__init__()
        self.network = network
        self.device = device
        self.time_step = nn.Parameter(torch.ones(1, dtype=torch.double), requires_grad=True)
        with torch.no_grad():
           self.time_step.data.copy_(time_step)

    def forward(self, input_batch):
      du = 11/6*input_batch[:,3].to(self.device) - 3*input_batch[:,2].to(self.device) + 3/2*input_batch[:,1].to(self.device) - 1/3*input_batch[:,0].to(self.device)
      ddu = 2*input_batch[:,3].to(self.device) - 5*input_batch[:,2].to(self.device) + 4*input_batch[:,1].to(self.device) - 1*input_batch[:,0].to(self.device)
      net_out = self.time_step*(self.network(torch.cat([input_batch[:, -1].to(self.device), du, ddu], dim=2).float()).double())
      return input_batch[:,-1].to(self.device) + du + 0.5*ddu + net_out.float()




class Loss_Singlestep(nn.Module):
    def __init__(self, model, batch_time, loss_func):
        super().__init__()
        self.model = model
        self.batch_time = batch_time
        self.loss_func = loss_func

    def forward(self, batch):        
        x_i = self.model(batch[:,0])
        loss = self.loss_func(x_i, batch[:,1])
        loss.backward()
        for i in range(2, self.batch_time):
            x_i = self.model(batch[:,i-1])
            loss = self.loss_func(x_i, batch[:,i])
            loss.backward()
            
        return loss

class Loss_Multistep(nn.Module):
    def __init__(self, model, batch_time, loss_func):
        super().__init__()
        self.model = model
        self.batch_time = batch_time
        self.loss_func = loss_func

    def forward(self, batch):        
        x_1 = self.model(batch[:,0])
        loss = self.loss_func(x_1, batch[:,1])
        x_i = x_1
        for i in range(2, self.batch_time):
            x_i = self.model(x_i)
            loss = loss + self.loss_func(x_i, batch[:,i])
        return loss

class Loss_Multistep_Test(nn.Module):
    def __init__(self, model, batch_time, loss_func):
        super().__init__()
        self.model = model
        self.batch_time = batch_time
        self.loss_func = loss_func

    def forward(self, batch):        
        x_i = self.model(batch[:,0])
        loss = self.loss_func(x_i, batch[:,1])
        for i in range(2, self.batch_time):
            x_i = self.model(x_i)
            loss = loss + self.loss_func(x_i, batch[:,i])
        return loss
