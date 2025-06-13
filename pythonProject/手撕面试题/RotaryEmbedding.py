# RoPE相对位置编码RotaryEmbedding
# x_m * W_q 中 x_m是第m行向量， W_q分解成列向量q1, q2, q2后面就不写了。x_m, q1都是hidden_size维度
# x_m * W_q = x_m * [q1, q2] = [x_m * q1, x_m * q2]，可以看做Q = x * W_q中的第m行

# x_n * W_k 中 x_n是第m行向量
# x_n * W_k = x_n * [q1, q2] = [x_n * k1, x_n * k2]，可以看做K= x * W_k中的第n行，即K的转置的第n列

# [x_m * q1, x_m * q2]和[x_n * k1, x_n * k2]做内积就是Q * K_T的第m行第n列。
# RoPE相对位置编码的作用就是希望第m行第n列的计算结果有m-n的信息。

