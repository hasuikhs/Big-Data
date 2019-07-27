package com.test;

import java.io.IOException;
import java.util.ArrayList;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.json.simple.JSONArray;
import org.json.simple.parser.JSONParser;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

/**
 * Servlet implementation class PredictServlet
 */
@WebServlet("/predict")
public class PredictServlet extends HttpServlet {
	private static final long serialVersionUID = 1L;

	private SavedModelBundle sModel = 
			SavedModelBundle.load("C:/python_ML/TFJavaAPI/SaveModel", "serve");
    /**
     * @see HttpServlet#HttpServlet()
     */
    public PredictServlet() {
        super();
        // TODO Auto-generated constructor stub
    }

	/**
	 * @see HttpServlet#doGet(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		// TODO Auto-generated method stub
		response.getWriter().append("Served at: ").append(request.getContextPath());
	}

	private int findMaxNumber(float[] value,int valueLength, int maxIdx){
        for(int i=0; i<valueLength; i++){
            if(value[i] > value[maxIdx]){
                maxIdx = i;
            }
        }
        return maxIdx;
    }
	/**
	 * @see HttpServlet#doPost(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		System.out.println("서버호출성공!");
		String pData = request.getParameter("pixelData");		
		response.setContentType("text/html;charset=UTF8");

		float testInput[][] = new float[1][784];
		
		ArrayList<String> list = new ArrayList<String>();     
		JSONParser parser = new JSONParser();
		try {
			Object obj = parser.parse(pData);
			JSONArray jsonArray = (JSONArray)obj;
			
			if (jsonArray != null) { 
				JSONArray innerArr = (JSONArray)jsonArray.get(0);
			   int len = innerArr.size();
			   for (int i=0;i<len;i++){ 				   
				   String cName = innerArr.get(i).getClass().getName();
				   if( cName.equals("java.lang.Long")) {
					   testInput[0][i] = ((Long)innerArr.get(i)).floatValue();
				   } else {
						testInput[0][i] = ((Double)innerArr.get(i)).floatValue();					   
				   }
			   } 
			} 			
		} catch(Exception e) {
			System.out.println(e);
		}
		
		int result = -1;

		try {
			Session sess = sModel.session();			    
			
					        
			Tensor<?> x = Tensor.create(testInput);    
	        //run the model and get the result
	        float[][] y = sess.runner()
	                .feed("x", x)
	                .fetch("h")
	                .run()
	                .get(0)
	                .copyTo(new float[1][10]);		                
	        		        
	        
	        float[] value = y[0];
	        int valueLength =10;
	        int maxIdx = 0;		                
	        
	        result = findMaxNumber(value, valueLength, maxIdx);
			
			response.getWriter().println(result);			
		} catch(Exception e) {
			System.out.println(e);
		}
	}

}
