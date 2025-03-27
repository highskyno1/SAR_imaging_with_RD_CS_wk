%{
    本代码用于对雷达的回波数据，利用RD算法~精确版本
    2025/3/27 11:50
%}
close all;
%% 数据读取
% 加载数据
echo1 = importdata('CDdata1.mat');
echo2 = importdata('CDdata2.mat');
% 将回波拼装在一起
echo = double([echo1;echo2]);
% 加载参数
para = importdata('CD_run_params.mat');
Fr = para.Fr;   % 距离向采样率
Fa = para.PRF;  % 方位向采样率
f0 = para.f0;   % 中心频率
Tr = para.Tr;   % 脉冲持续时间
R0 = para.R0;   % 最近点斜距
Kr = -para.Kr;  % 线性调频率
c = para.c;     % 光速
% 以下参数来自课本附录A
Vr = 7062;      % 等效雷达速度
Ka = 1733;      % 方位向调频率
f_nc = -6900;   % 多普勒中心频率
lamda = c/f0;   % 波长

%% 图像填充
% 计算参数
[Na,Nr] = size(echo);
% 按照全尺寸对图像进行补零
echo = padarray(echo,[round(Na/6), round(Nr/3)]);
% 计算参数
[Na,Nr] = size(echo);

%% 轴产生
% 距离向时间轴及频率轴
tr_axis = 2*R0/c + (-Nr/2:Nr/2-1)/Fr;   % 距离向时间轴
fr_gap = Fr/Nr;
fr_axis = fftshift(-Nr/2:Nr/2-1).*fr_gap;   % 距离向频率轴

% 方位向时间轴及频率轴
ta_axis = (-Na/2:Na/2-1)/Fa;        % 方位向时间轴
ta_gap = Fa/Na; 
fa_axis = f_nc + fftshift(-Na/2:Na/2-1).*ta_gap;    % 方位向频率轴
% 方位向对应纵轴，应该转置成列向量
ta_axis = ta_axis';
fa_axis = fa_axis';

%% 第一步 距离压缩
% 方位向下变频
echo_s1 = echo .* exp(-2i*pi*f_nc.*ta_axis);
% 距离向傅里叶变换
echo_s1 = fft(echo_s1,[],2);
% 距离向距离压缩滤波器
echo_d1_mf = exp(1i*pi/Kr.*fr_axis.^2);
% 距离向匹配滤波
echo_s1 = echo_s1 .* echo_d1_mf;
% 回到时域
echo_s1 = ifft(echo_s1,[],2);

%% 第二步 方位向傅里叶变换&距离徙动矫正
% 方位向傅里叶变换
echo_s2 = fft(echo_s1,[],1);
% 计算徙动因子
R_trix = tr_axis*c/2;
D = lamda^2/8/Vr^2.*fa_axis.^2.*(R_trix);
% 插值校正
foo = zeros(size(echo_s2));
for i = 1:Na
    foo(i,:) = interp1(R_trix, echo_s2(i,:),R_trix+D(i,:),"spline",0);
end
echo_s2 = foo;

%% 第三步 方位压缩
% 方位向滤波器
echo_d3_mf = exp(-1i*pi/Ka.*fa_axis.^2);
% 方位向脉冲压缩
echo_s3 = echo_s2 .* echo_d3_mf;
% 方位向逆傅里叶变换
echo_s3 = ifft(echo_s3,[],1);

%% 数据最后的矫正
% 根据实际观感，方位向做合适的循环位移
echo_s4 = circshift(abs(echo_s3), -3193, 1);
% 上下镜像
echo_s4 = flipud(echo_s4);
echo_s5 = abs(echo_s4);
saturation = 50;
echo_s5(echo_s5 > saturation) = saturation;

%% 成像
% 绘制处理结果热力图
figure;
imagesc(tr_axis.*c,ta_axis.*c,echo_s5);
title('处理结果(RD算法)');
% 以灰度图显示
echo_res = gather(echo_s5 ./ saturation);
% 直方图均衡
echo_res = adapthisteq(echo_res,"ClipLimit",0.004,"Distribution","exponential","Alpha",0.5);
figure;
imshow(echo_res);
