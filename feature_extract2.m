%**************************************************************************
%�������ƣ�feature_extract2()
%������A:����ͼ������
%����ֵ��feature��ͼ������ֵ
%�������ܣ�������ȡ
%*************************************************************************a
function feature=feature_extract2(A)
%����������*****************************************************************
[r,c]=size(A);                          %r�������������c�����������
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

%��������ƽ����Ϊ5*5��С���򣬲�����ÿ��С�����а�������ռ��������һ�е�5������ֵ
%���浽����ֵ��ǰ5�����ڶ��ж�Ӧ������ֵ��6~10�����Դ�����*********************
e=mod(r,5);
A=A';
if e==1                                 %ͨ���ʵ����������ʹ�����������Ա�5����
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
B=zeros(r,c);                           %��ֵ��
for i=1:r                     
    for j=1:c
        if A(i,j)>0
            B(i,j)=1;
        end
    end
end
        
a=r/5;                                  %������ȡ
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
        feature(5*(i-1)+j)=sum/(m*n);   %����ֵ
    end
end
    
    
            
    







































        
    
