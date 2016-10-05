training_data_indicies = new Array(18).fill(0);
initial_image_indicies = new Array(4).fill(0);

$(function() {                       //run when the DOM is ready
  $(".thumbnail").click(function() {  //use a class, since your ID gets mangled
    $(this).toggleClass("thumbnail-clicked");      //add the class to the clicked element
  });
});

$('a.thumbnail').removeAttr('href');
$('#train-and-draw').removeAttr('href');

$(function() {
	$("#train-and-draw").click(function() {

		for (var i = 0; i < training_data_indicies.length; i++) {
	    	if ($("#tdata"+String(i)).hasClass("thumbnail-clicked")) {
	    		training_data_indicies[i] = 1;
	    	} else {
	    		training_data_indicies[i] = 0;
	    	}			
		}

		for (var i = 0; i < initial_image_indicies.length; i++) {
			if ($("#iimage"+String(i)).hasClass("thumbnail-clicked")) {
				initial_image_indicies[i] = 1;
			} else {
				initial_image_indicies[i] = 0;
			}
		}

    	console.log(training_data_indicies);
    	console.log(initial_image_indicies);

    	data_to_python = {"training_data_indicies": training_data_indicies, 
    					  "initial_image_indicies": initial_image_indicies};

    	d3.json('/run/').post(JSON.stringify(data_to_python), function(error, data) {
    			console.log(data);
    			console.log(error);
    			// row = d3.select("#results").insert("div").attr("class", "row");
    			// row.append("div").attr("class", "col-md-6").attr("style", "padding: 0px")
    			//    .append("div").attr("class", "col-md-6").attr("style", "padding: 0px")
    			//    .append("div").attr("class", "col-xs-3 col-md-6")
    			//    .append("a").attr("href", "#").attr("class", "thumbnail thumbnail-result")
    			//    .append("img").attr("src", "results/"+"1"+".png")
    			//    .data(data.errors).enter().append("div").attr("class", "caption error-metric").text(function(d) {return d});

    			console.log("made result row");
    		}
		);
	})
})

