$(function() {                       //run when the DOM is ready
  $(".thumbnail").click(function() {  //use a class, since your ID gets mangled
    $(this).toggleClass("thumbnail-clicked");      //add the class to the clicked element
  });
});

$('a.thumbnail').removeAttr('href');