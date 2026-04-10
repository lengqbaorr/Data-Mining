# BAO CAO NOI DUNG DA THUC HIEN VA Y NGHIA PHAN TICH

## Tong quan notebook
- Ten notebook: EDA_NASA_Turbofan.ipynb
- Tong so cell: 69
- Muc tieu xuyen suot: khong chi mo ta du lieu, ma bien du lieu thanh co so de huan luyen mo hinh du doan RUL on dinh va co the giai thich.

## Cach doc bao cao nay
Moi cum cell duoc trinh bay theo 3 cau hoi:
1. Da lam gi?
2. Vi sao lam nhu vay?
3. Rut ra duoc gi cho bai toan RUL?

## 1. Khoi dong va nap du lieu (Cell 1-4)
### Da lam gi
- Cell 1-2: Dat boi canh bai toan, mo ta 4 bo FD001-004 va cau truc cot.
- Cell 3: Import thu vien, cau hinh hien thi/plot.
- Cell 4: Tao schema cot, nap dong bo train/test/RUL cho 4 bo du lieu voi co che tim duong dan linh hoat.

### Vi sao lam nhu vay
- Can chuan hoa ten cot ngay tu dau de cac buoc sau (groupby, corr, feature engineering) khong sai schema.
- Co che tim duong dan linh hoat giup notebook chay duoc trong nhieu moi truong khac nhau.

### Rut ra duoc gi
- Du lieu da san sang cho kiem dinh va EDA.
- Giam rui ro loi IO/duong dan, tranh mat thoi gian debug khong lien quan den nghiep vu.

## 2. Data Validation (Cell 5-18)
### 2.1 Kich thuoc va tinh dong nhat (Cell 5-7)
### Da lam gi
- Tao bang tong quan so dong, so cot, so engine train/test va so dong nhan RUL.
- Doi chieu test engines voi RUL count.

### Vi sao lam nhu vay
- Neu so luong engine test khong khop so nhan RUL, toan bo danh gia mo hinh sau nay se sai mapping.

### Rut ra duoc gi
- Xac nhan tinh nhat quan cap dataset-label truoc khi phan tich sau.

### 2.2 Missing values (Cell 8-9)
### Da lam gi
- Quet missing tren tat ca train/test cua 4 bo.

### Vi sao lam nhu vay
- Missing values co the gay lech thong ke, vo hieu hoa corr, va gay fail khi train model.

### Rut ra duoc gi
- Khi khong co missing, co the bo qua imputation va giu pipeline gon nhe hon.

### 2.3 Duplicate rows va duplicate key (Cell 10-11)
### Da lam gi
- Kiem tra duplicate dong va duplicate cap (unit_id, cycle).

### Vi sao lam nhu vay
- Duplicate khoa thoi gian theo engine se lam sai trajectory suy hao va lam model hoc lap lai mau.

### Rut ra duoc gi
- Xac nhan chuoi thoi gian moi engine khong bi trung lap bat thuong.

### 2.4 Data types va phan bo co ban (Cell 12-14)
### Da lam gi
- Kiem tra dtype, xac nhan tat ca cot la numeric.
- Xem descriptive stats tren FD001.

### Vi sao lam nhu vay
- Numeric-toan-bo la dieu kien can de tinh corr, rolling, scaling, va train model.
- Thong ke co ban giup phat hien bat thuong ve bien do gia tri.

### Rut ra duoc gi
- Du lieu dat dieu kien ky thuat de xu ly tiep ma khong can encode.

### 2.5 Constant/near-constant (Cell 15-16)
### Da lam gi
- Tinh do lech chuan cho sensor + op_setting, tach nhom constant va near-constant.

### Vi sao lam nhu vay
- Feature gan nhu khong doi thuong khong mang thong tin du doan, co the tang nhieu va on.

### Rut ra duoc gi
- Co danh sach ung vien can loai bo de giam chieu va tang do on dinh mo hinh.

### 2.6 Cycle continuity (Cell 17-18)
### Da lam gi
- Kiem tra chu ky tung engine co lien tuc 1..N hay khong.

### Vi sao lam nhu vay
- Pipeline time-series (EMA, rolling, trend) gia dinh dong thoi gian lien tuc.

### Rut ra duoc gi
- Co so hop le de trich xuat dac trung dong hoc.

## 3. EDA chi tiet (Cell 19-46)
### 3.1 Tao nhan RUL train (Cell 19-21)
### Da lam gi
- Tinh RUL = max_cycle - cycle cho tung unit_id.

### Vi sao lam nhu vay
- Day la cach label chuan cho du lieu run-to-failure khi train khong co cot RUL san.

### Rut ra duoc gi
- Co target supervision de phan tich lien he sensor-RUL va train model.

### 3.2 Phan bo tuoi tho engine (Cell 22-23)
### Da lam gi
- Ve histogram max cycle/engine cho 4 bo, kem mean/median va thong ke.

### Vi sao lam nhu vay
- De biet muc do bien thien tuoi tho giua engine va giua cac bo FD.

### Rut ra duoc gi
- Nhan ra bai toan co tinh khong dong nhat ve tuoi tho, anh huong truc tiep den do kho du doan.

### 3.3 Phan bo RUL test (Cell 24-25)
### Da lam gi
- Ve histogram ground-truth RUL cua test cho moi FD.

### Vi sao lam nhu vay
- Hieu phan bo nhan danh gia giup chon metric va hieu loi du doan theo tung vung RUL.

### Rut ra duoc gi
- Co cai nhin ve do lech phan bo test, tranh danh gia mo hinh theo mot goc nhin don le.

### 3.4 Operational settings (Cell 26-27)
### Da lam gi
- Ve scatter op settings va uoc tinh so cum dieu kien van hanh.

### Vi sao lam nhu vay
- Dieu kien van hanh la bien gay shift phan bo, dac biet quan trong voi FD002/FD004.

### Rut ra duoc gi
- Xac nhan FD001/FD003 don gian hon (it regime), FD002/FD004 phuc tap hon (nhieu regime).

### 3.5 Tong quan sensor va variance (Cell 28-30)
### Da lam gi
- Chuan hoa de ve boxplot so sanh sensor.
- Phan tich variance, tach useful_sensors va constant_sensors.

### Vi sao lam nhu vay
- Sensor khac don vi can dua ve cung thang de so sanh cong bang.
- Variance thap giup nhan dien bien kem thong tin.

### Rut ra duoc gi
- Co bo sensor uu tien cho mo hinh, va bo sensor nen loai de giam nhieu.

### 3.6 Degradation theo thoi gian (Cell 31-32)
### Da lam gi
- Ve duong sensor theo cycle tren nhieu engine mau.

### Vi sao lam nhu vay
- Can xac thuc bang mat rang sensor co quy luat suy hao theo thoi gian.

### Rut ra duoc gi
- Nhieu sensor the hien xu huong suy giam ro, ung ho gia thuyet du doan RUL la kha thi.

### 3.7 Sensor theo RUL (Cell 33-34)
### Da lam gi
- Ve scatter sensor-RUL va trend mean theo RUL.

### Vi sao lam nhu vay
- Truc tiep do suc manh lien he giua feature va target.

### Rut ra duoc gi
- Xac dinh nhom sensor co quan he don dieu/ro net voi RUL, phu hop cho feature selection.

### 3.8 Correlation va top feature (Cell 35-38)
### Da lam gi
- Ve heatmap tuong quan toan bo feature + RUL.
- Xep hang top tuong quan voi RUL theo |r|.
- Giai thich tai sao dung tri tuyet doi.

### Vi sao lam nhu vay
- Bai toan du doan quan tam do manh lien he, khong quan tam huong am/duong theo nghia loai bo.

### Rut ra duoc gi
- Co danh sach ung vien feature manh cho mo hinh baseline.

### 3.9 Multicollinearity (Cell 39-40)
### Da lam gi
- Tim cap sensor co |r| > 0.9.

### Vi sao lam nhu vay
- Da cong tuyen lam mo hinh de bi overfit va giam kha nang dien giai.

### Rut ra duoc gi
- Co co so de ap dung giam chieu/PCA/feature pruning.

### 3.10 Outlier (Cell 41-42)
### Da lam gi
- Dung IQR dem ty le outlier moi sensor va truc quan hoa.

### Vi sao lam nhu vay
- Outlier co the lam meo scaler, corr va mo hinh tree/linear theo cach khac nhau.

### Rut ra duoc gi
- Xac dinh sensor can robust scaling, winsorize hoac xu ly dac thu.

### 3.11 Cross-dataset comparison (Cell 43-44)
### Da lam gi
- So sanh phan bo mot so sensor tren ca 4 bo.

### Vi sao lam nhu vay
- De danh gia kha nang chuyen giao mo hinh giua cac bo du lieu.

### Rut ra duoc gi
- Co dau hieu domain shift giua cac FD, can chien luoc training rieng hoac chuan hoa theo regime.

### 3.12 Deep dive 1 engine (Cell 45-46)
### Da lam gi
- Ve full lifecycle cua engine #1 voi cac sensor con thong tin.

### Vi sao lam nhu vay
- Kiem tra cap do vi mo xem quy luat suy hao co ro tren mot ca the cu the hay khong.

### Rut ra duoc gi
- Xac nhan hanh vi suyt hao theo trajectory, ho tro tu duy mo hinh theo chuoi thoi gian.

## 4. Tong ket EDA va dien giai bo sung (Cell 47-49)
### Da lam gi
- Cell 47-48: Tong hop ket qua validation + EDA + khuyen nghi modeling.
- Cell 49: Dien giai bang tieng Viet ve bo du lieu va nhung insight chinh.

### Vi sao lam nhu vay
- Chuyen tu ket qua ky thuat sang ket luan nghiep vu co the hanh dong.

### Rut ra duoc gi
- Dinh huong ro cho buoc tien xu ly va huan luyen mo hinh tiep theo.

## 5. Data processing nang cao cho FD001 va FD003 (Cell 50-67)
### 5.1 Loc feature theo do bien dong + corr (Cell 50-54)
### Da lam gi
- Loai cot constant.
- Xu ly near-constant bang test corr voi cycle de quyet dinh xoa/giu.

### Vi sao lam nhu vay
- Khong xoa may moc toan bo near-constant, tranh mat feature co gia tri xu huong.

### Rut ra duoc gi
- Giu duoc feature "it bien dong nhung co y nghia", dong thoi giam feature nhieu vo nghia.

### 5.2 Piecewise RUL (Cell 55-57)
### Da lam gi
- Tao nhan RUL va cat tran tai MAX_RUL = 125.

### Vi sao lam nhu vay
- Giai doan dau vong doi thuong it thong tin ve hu hong, cat tran giup mo hinh tap trung vung suy giam co y nghia.

### Rut ra duoc gi
- Nhan train on dinh hon, giam anh huong cua vung "qua som" den hoc may.

### 5.3 EMA smoothing (Cell 58-60)
### Da lam gi
- Lam muot sensor theo tung engine bang EMA span=10.

### Vi sao lam nhu vay
- Giam nhieu tan so cao de lam ro xu huong suy hao.

### Rut ra duoc gi
- Tin hieu dau vao sach hon cho rolling/trend feature.

### 5.4 Feature engineering dong hoc (Cell 61-63)
### Da lam gi
- Tao roll_mean, roll_std, roll_min, roll_max, trend cho moi sensor.

### Vi sao lam nhu vay
- RUL phu thuoc khong chi gia tri tuc thoi ma con dong luc bien doi theo cua so thoi gian.

### Rut ra duoc gi
- Mo hinh co them thong tin ve muc nen, bien dong va toc do suy hao.

### 5.5 Scaling (Cell 64-66)
### Da lam gi
- MinMax scaling tren train, ap dung lai cho test.

### Vi sao lam nhu vay
- Dua cac feature ve cung mien gia tri, tranh feature bien do lon ap dao feature khac.

### Rut ra duoc gi
- Tang tinh on dinh khi train va giup so sanh feature cong bang hon.

### 5.6 Dinh hinh du lieu dau vao 2D/3D (Cell 67)
### Da lam gi
- Giu bo cot cuoi cung cho model.
- Tao test_last de danh gia 2D.
- Luu test_full va ham gen_sequence cho huong LSTM 3D.

### Vi sao lam nhu vay
- Mot pipeline can ho tro ca model tabular va model chuoi.

### Rut ra duoc gi
- Du lieu da san sang cho nhieu nhanh mo hinh khac nhau ma khong can tien xu ly lai tu dau.

## 6. Baseline modeling Random Forest (Cell 68)
### Da lam gi
- Train RandomForestRegressor cho FD001 va FD003.
- Danh gia bang RMSE tren test.
- In top 5 feature importance.

### Vi sao lam nhu vay
- Can mot baseline manh, de train va de dien giai, truoc khi qua model phuc tap hon.

### Rut ra duoc gi
- Co moc tham chieu hieu nang ban dau va biet feature nao dong gop nhieu nhat.

## 7. Truc quan ket qua du doan (Cell 69)
### Da lam gi
- Ve duong RUL thuc te vs du doan theo cycle cho 1 engine cu the.
- Ve so sanh RUL thuc te vs du doan tren toan bo engine test.

### Vi sao lam nhu vay
- RMSE chi cho mot con so tong hop; can do thi de thay mo hinh sai o dau, sai theo kieu gi.

### Rut ra duoc gi
- Danh gia duoc ca chat luong tong quan va hanh vi du doan theo quy trinh suy hao.

## Ket luan tong hop
- Notebook da thuc hien day du vong doi: validation -> EDA -> tao nhan -> lam sach -> lam muot -> tao dac trung -> scale -> train baseline -> danh gia truc quan.
- Y nghia lon nhat khong nam o viec "ve nhieu bieu do", ma o cho moi buoc deu giai quyet mot rui ro cu the trong bai toan RUL:
1. Rui ro du lieu loi va sai mapping nhan.
2. Rui ro feature khong thong tin hoac da cong tuyen.
3. Rui ro bo qua dong hoc suy hao theo thoi gian.
4. Rui ro danh gia mo hinh chi bang 1 metric.

## Goi y buoc tiep theo
1. Lam pipeline rieng cho FD002/FD004 theo operating regime.
2. Thu mo hinh sequence (LSTM/GRU/Transformer) tren du lieu da co ham cat sequence.
3. Thu robust scaling/clip outlier truoc khi train de so sanh voi baseline hien tai.
