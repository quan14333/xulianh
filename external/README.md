# Mã nguồn bên ngoài

`external/` dành cho upstream được pin theo commit/tag, ví dụ SAGE_IVIF. Không sao chép
âm thầm mã upstream vào `src/`. Ghi URL, commit, license và mọi patch cục bộ. Có thể clone
cục bộ vào đây nhưng chỉ commit khi quyền phân phối và chiến lược vendor đã được xác nhận.

`src/` chỉ chứa integration, evaluation và phương pháp do nhóm duy trì.
