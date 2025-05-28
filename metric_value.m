close all;
clear;
label_path='D:\Users\Administrator\PycharmProjects\US\home\PA_US\result\12_09_UIU-Net_ft_3_4\labels';
predict_path='D:\Users\Administrator\PycharmProjects\US\home\PA_US\result\12_09_UIU-Net_ft_3_4\predictions';
dir0='*.mat';
lable_dir=fullfile(label_path,dir0);
predict_dir=fullfile(predict_path,dir0);
label_dir1=dir(lable_dir);
predict_dir1=dir(predict_dir);
label_fileNames = {label_dir1.name};
predict_fileNames = {predict_dir1.name};
label_fileNames = sort_nat(label_fileNames);
predict_fileNames=sort_nat(predict_fileNames);
av_AC2=0;
av_AC=0;
av_AC1=0;
av_SE=0;
av_F1=0;
av_JS=0;
av_DC=0;
av_dice=0;
av_SP=0;
av_PC=0;
av_num=0;
for i=1:size(predict_fileNames,2)
    label = load(fullfile(label_path,label_fileNames{i})).data;
    predict = load(fullfile(predict_path,predict_fileNames{i})).data;
    % predict = im2double(predict);
    % label = im2double(label);
    TN=0;
    TP=0;
    FN=0;
    FP=0;
    lb=0;
    pd=0;
    for j=1:size(label,1)
        for k=1:size(label,2)
            if label(j,k)==1 && predict(j,k)==1
                TP=TP+1;
                lb=lb+1;
                pd=pd+1;
            elseif label(j,k)==1 && predict(j,k)==0
                FN=FN+1;
                lb=lb+1;
            elseif label(j,k)==0 && predict(j,k)==0
                TN=TN+1;
            else
                FP=FP+1;
                pd=pd+1;
        end
        end
    end
    % A = rgb2dec(A);
    AC=double((TP+TN)/(TP+TN+FP+FN)); %准确率
    AC1=double(TP/lb); %召回率
    AC2=double(TP/pd);   %精确度
    SE=double(TP/(TP+FN));
    SP=double(TN/(TN+FP));
    PC=double(TP/(TP+FP));
    F1=double(2*((PC*SE)/(PC+SE)));
    JS=double(TP/(TP+FN+FP));
    DC=double((2*TP)/(2*TP+FP+FN));
    similarity = dice(label, predict);%调用dice函数
    % if AC2>0.7
    av_num=av_num+1;
    av_AC2=av_AC2+AC2;
    av_AC1=av_AC1+AC1;
    av_AC=av_AC+AC;
    av_SE=av_SE+SE;
    av_F1=av_F1+F1;
    av_JS=av_JS+JS;
    av_DC=av_DC+DC;
    av_dice=av_dice+similarity;
    av_SP=av_SP+SP;
    av_PC=av_PC+PC;
    fprintf('\n%s.jpg dice:%.3f AC:%.3f AC1:%.3f AC2:%.3f SE:%.3f SP:%.3f PC:%.3f F1:%.3f JS:%.3f DC:%.3f',num2str(i),similarity,AC,AC1,AC2,SE,SP,PC,F1,JS,DC);
    % end
    end
    
    av_AC2=av_AC2/av_num;
    av_AC1=av_AC1/av_num;
    av_AC=av_AC/av_num;
    av_SE=av_SE/av_num;
    av_F1=av_F1/av_num;
    av_JS=av_JS/av_num;
    av_DC=av_DC/av_num;
    av_dice=av_dice/av_num;
    av_SP=av_SP/av_num;
    av_PC=av_PC/av_num;
    fprintf('\n av_AC2:%.3f av_AC1:%.3f av_AC:%.3f av_SE:%.3f av_F1:%.3f av_JS:%.3f av_DC:%.3f av_dice:%.3f av_SP:%.3f av_PC:%.3f ',av_AC2,av_AC1,av_AC,av_SE,av_F1,av_JS,av_DC,av_dice,av_SP,av_PC);
    % 
    function d = dice (img1 ,img2)
    intersection = sum(img1(:) & img2(:)); %交集像素数
    union = sum (img1(:) | img2(:));%并集像素数
    d = double(2 * intersection / union) ; %dice系数
end

