// drop된 이미지를 28x28 이미지로 만들고
// pixel data를 얻어오는 것이 목적
function imgDrop(){
	event.preventDefault();
	
	// drop된 이미지의 reference를 획득(HTML5)
	var f = event.dataTransfer.files[0];
	
	// drop된 파일의 내용을 이용하여 새로운 이미지를 만듦
	var newImg = new Image();
	newImg.width = 200; // 200x200짜리 비어있는 이미지 객체
	var resizeImg = new Image();
	resizeImg.width = 28;
	
	// 파일로 부터 데이터를 읽어들이기 위한 FileReader를 생성
	var imageReader = new FileReader();
	
	// 비동기 이벤트를 위한 처리
	imageReader.onload = function(){
		
		resizeImg.onload = function(){
			// pixel data를 뽑아내기 위한 용도
			// HTML5에서 제공하는 canvas를 생성해서 canvas에 이미지를 그림
			var myCanvas = document.createElement("canvas");
			myCanvas.width = 28;
			myCanvas.height = 28;
			var ctx = myCanvas.getContext("2d");
			ctx.drawImage(resizeImg, 0, 0, 28, 28);
			
			// 이렇게 canvas에 그림을 그리면 canvas로부터 pixel data 추출가능
			idata = ctx.getImageData(0, 0, 28, 28);
			// idata 안에는 3가지 정보가 들어가 있음
			// 가로길이, 세로길이, 픽셀에 대한 배열,
			// idata.data => pixel정보를 가지고 있는 배열
			// [0, 23, 12, 0, 0, 1, 3, 4, 0, 66, 23, 42, 1, 0, 23, 43, 23]
			// 4개가 1개의 픽셀정보를 나타냄
			// rgba 값으로 1개의 픽셀이 4개의 값으로 표현
			// 흑백이미지가 필요하고 [1][784] 형태의 이미지가 필요
			// 일단 grey형태로 이미지를 변형
			var result = new Array(new Array(1), new Array(784));
			var count = 0;
			for(var i = 0; i < idata.data.length; i += 4){
				var avg = (idata.data[i] + idata.data[i + 1] + idata.data[i + 2]) / 3.0; 
				result[0][count++] = avg;
				// 0 -> 가장 밝은색, 255 -> 가장 어두운 색
				// min max scale형태로 바꾸어서 저장해야 함
			}
			
			console.log(result);
		}
		
		newImg.onload = function(){
			$("#targetDiv").append(newImg);
		}
		
		newImg.src = event.target.result;
		resizeImg.src = event.target.result;
	}
	
	imageReader.readAsDataURL(f);

}
