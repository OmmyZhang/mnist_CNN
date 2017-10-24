import numpy as np

def conv2d_forward(input, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_out (#output channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after convolution
    '''
#    print 'f:',input.shape


    n = input.shape[0]
    c_out = W.shape[0]

#    print input.shape, W.shape, kernel_size
#    print len(W) , '=', len(b)

    input = np.pad(input,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')

    col_A, h_out, w_out = im2col_A(input, kernel_size)
    col_K = im2col_K(W, kernel_size)
    
    return np.transpose(np.dot(col_A, col_K),(0,2,1)).reshape((n, c_out, h_out, w_out)) + b.repeat(h_out * w_out).reshape((c_out,h_out,w_out))


def conv2d_backward(input, grad_output, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_out (#output channel) x h_out x w_out
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_W: gradient of W, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        grad_b: gradient of b, shape = c_out
    '''
#    print 'W:',W.shape
#    print 'g_out:',grad_output.shape

    c_in = W.shape[1]
    c_out = W.shape[0]
    w_out = grad_output.shape[3]
    
    input = np.pad(input,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')
    
    grad_input = conv2d_forward(grad_output, np.rot90(W.transpose((1,0,2,3)), 2, (2,3)), np.zeros(c_in), kernel_size, kernel_size - 1)

    if(pad > 0):
        grad_input = grad_input[:,:, pad:-pad, pad:-pad]

    grad_W = conv2d_forward(input.transpose((1,0,2,3)),grad_output.transpose((1,0,2,3)), np.zeros(c_out), w_out , 0).transpose(1,0,2,3) 
    # NOTE:Only for  h_out = w_out. It's enough for this Homework.
    grad_b = grad_output.sum(axis=(0,2,3))

    return grad_input, grad_W, grad_b


def avgpool2d_forward(input, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_in (#input channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after average pooling over input
    '''
    n    = input.shape[0]
    c_in = input.shape[1]
    h_in = input.shape[2]
    w_in = input.shape[3]

    input = input.reshape(n*c_in, 1, h_in, w_in)

    input = np.pad(input,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')
    col_A, h_out, w_out = im2col_A(input,kernel_size,kernel_size)

    return col_A.mean(2).reshape(n,c_in,h_out,w_out)
    

def avgpool2d_backward(input, grad_output, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_in (#input channel) x h_out x w_out
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
    '''
    grad_in = grad_output.repeat(kernel_size,2).repeat(kernel_size,3) / (kernel_size * kernel_size)
#    print 'pooling_back:',grad_in.shape
    if(pad > 0):
        grad_in = grad_in[:, :, pad : -pad, pad : -pad] 
    return grad_in
    
    
def im2col_A(A, k_size, stride=1):
    n    = A.shape[0]
    c_in = A.shape[1]

    y = (A.shape[2] - k_size) / stride + 1
    x = (A.shape[3] - k_size) / stride + 1
    
    k2 = k_size * k_size
    R = np.empty((n, x * y, k2 * c_in))

    for ci in xrange(c_in):
        for i in xrange(y):  
            for j in xrange(x):
                R[:, i * x + j, ci * k2:(ci+1) * k2] = A[:, ci, i * stride:i * stride + k_size, j * stride:j * stride + k_size].reshape(n,-1)
    return R, y, x 

def im2col_K(K, k_size):
    c_out = K.shape[0]
    c_in  = K.shape[1]

    k2 = k_size * k_size
    R = np.empty((k2 * c_in, c_out))

    for i in xrange(c_out):
        for j in xrange(c_in):
            R[j * k2:(j+1) * k2, i] = K[i][j].ravel().T #[::-1].T
    return R


'''
# n = 3,c_in = 2, h_in = h_out = 4
# c_out = 3
kernel_size = 2
pad = 1

input = np.reshape(range(1,97),(3,2,4,4))
W =  np.reshape(range(1,25),(3,2,2,2));
b = np.ones(3);

print input

out =  conv2d_forward(input, W, b, 2, 1)
conv2d_backward(input, out, W, b, 2, 1)
#print avgpool2d_forward(input,3,1)


inp = np.reshape(range(36),(1,1,6,6))
W = np.reshape([0,0,-1,1,-1,1,-1,1,1],(1,1,3,3))
g_o = np.reshape([0,0,1,2,2,2,0,0,2,1,2,2,3,0,1,1],(1,1,4,4))
k_s = 3

g_i, g_w, g_b = conv2d_backward(inp, g_o, W, np.arange(1), k_s, 0)
print g_i
'''
