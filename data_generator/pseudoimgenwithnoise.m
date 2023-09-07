clear;

k = [1, 0.189, 0.189*2, 0.189*4, 0.189/2, 0.189/5, 0.189/10, 0.189/100];

for fidx = 1:6
    folderpath = ['D:\JRC-work\PROJECTs\DL-Enhanced SRS\20230906-noisebatch',...
        '\group_',num2str(fidx-1)];
    cd(folderpath);
    lp_length = round(32*2^(fidx-1));
    for j = 1:1:length(k)
        subfname = num2str(k(j));
        mkdir(subfname);
        cd(subfname);
        for i = 1:1:100
            img_temp = caimgen(lp_length,1024);
            img_temp2 = double(imnoise(uint16(img_temp),'speckle'));
            img_blur_temp = blurgen(img_temp2,k(j));
            img_gt = imcrop(img_temp,[1024 1 1023 1023]);
            img_gtn = imcrop(img_temp2,[1024 1 1023 1023]);
            img_blur = imcrop(img_blur_temp,[1024 1 1023 1023]);
            imwrite(uint16(img_gt),...
                ['gt_k=',num2str(k(j)),'_',num2str(i),'.tif']);
            imwrite(uint16(img_gtn),...
                ['gtn_k=',num2str(k(j)),'_',num2str(i),'.tif']);
            imwrite(uint16(img_blur),...
                ['k=',num2str(k(j)),'_',num2str(i),'.tif']);
        end
        cd(folderpath);
    end
end



function imgraw = caimgen(lp_length,imgsize)

% cellular automata generates a mask for large landscape pattern
% mask = elementaryCellularAutomata(randi(255),lp_length,lp_length,rand()); 

mask = elementaryCellularAutomata(randi(255),lp_length,lp_length,rand());
pix_value = round(rand(lp_length)*4095);

imgseed = mask.*pix_value;
imgraw = imresize(imgseed,[imgsize*2 imgsize*2]);
imgraw = imcrop(imgraw,[1 imgsize imgsize*2-1 imgsize-1]);

end

function img_gtf = blurgen(img_gt,k)

dim = size(img_gt);

parfor i = 1:dim(1)
    r = filter([0 k], [1 -(1-k)],img_gt(i,:));
    img_gtf(i,:) = real(r);
end
end