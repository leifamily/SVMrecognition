%**************************************************************************
%函数名称：feature_extract2()
%参数：A:读入图像数据
%返回值：feature：图像特征值
%函数功能：特征提取
%*************************************************************************a
function feature=feature_extract2(A)
%搜索数据区*****************************************************************
[r,c]=size(A);                          %r：矩阵的行数；c：矩阵的列数
for i=r:-1:1
    if A(i,:)==zeros(1,c)
        A(i,:)=[];
    end
end
[r,c]=size(A);
for i=c:-1:1
    if A(:,i)==zeros(r,1)
        A(:,i)=[];
    end
end
[r,c]=size(A);

%将数据区平均分为5*5的小区域，并计算每个小区域中白像素所占比例，第一行的5个比例值
%保存到特征值的前5个，第二行对应着特征值的6~10个，以此类推*********************
e=mod(r,5);
A=A';
if e==1                                 %通过适当的添加行列使得数据区可以被5整除
    A=[zeros(c,2),A,zeros(c,2)];
elseif e==2
    A=[zeros(c,1),A,zeros(c,2)];
elseif e==3
    A=[zeros(c,1),A,zeros(c,1)];
elseif e==4
    A=[A,zeros(c,1)];
end
A=A';
[r,c]=size(A);
f=mod(c,5);
if f==1
    A=[zeros(r,2),A,zeros(r,2)];
elseif f==2
    A=[zeros(r,1),A,zeros(r,2)];
elseif f==3
    A=[zeros(r,1),A,zeros(r,1)];
elseif f==4
    A=[A,zeros(r,1)];
end

[r,c]=size(A);
B=zeros(r,c);                           %二值化
for i=1:r                     
    for j=1:c
        if A(i,j)>0
            B(i,j)=1;
        end
    end
end
        
a=r/5;                                  %特征提取
b=c/5;
feature=zeros(1,25);
for i=1:5
    for j=1:5 
        C=B(a*(i-1)+1:1:a*i,b*(j-1)+1:1:b*j);
        sum=0;
        for m=1:a
            for n=1:b
                sum=sum+C(m,n);
            end
        end
        feature(5*(i-1)+j)=sum/(m*n);   %返回值
    end
end
    
    
            
    







































        
    
