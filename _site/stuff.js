

// Button to get to the top of the page
// Get the button:
let mybutton = document.getElementById("myBtn");

// When the user scrolls down 20px from the top of the document, show the button
window.onscroll = function() {scrollFunction()};

function scrollFunction() {
  if (document.body.scrollTop > 300 || document.documentElement.scrollTop > 300) {
    mybutton.style.display = "block";
  } else {
    mybutton.style.display = "none";
  }
}

// When the user clicks on the button, scroll to the top of the document
function topFunction() {
    document.body.scrollTop = 0; // For Safari
    document.documentElement.scrollTop = 0; // For Chrome, Firefox, IE and Opera
  }



let processScroll = () => {
  let docElem = document.documentElement, 
    docBody = document.body,
    scrollTop = docElem['scrollTop'] || docBody['scrollTop'],
      scrollBottom = (docElem['scrollHeight'] || docBody['scrollHeight']) - window.innerHeight,
    scrollPercent = scrollTop / scrollBottom * 100 + '%';
    
  // console.log(scrollTop + ' / ' + scrollBottom + ' / ' + scrollPercent);
  
    document.getElementById("progress-bar").style.setProperty("--scrollAmount", scrollPercent); 
}
  
document.addEventListener('scroll', processScroll);
  



// $(document).scroll(function () {
//     $('section').each(function () {
//         if($(this).position().top <= $(document).scrollTop() && ($(this).position().top + $(this).outerHeight()) > $(document).scrollTop()) {
            
//             console.log($(this).attr('id'));
//         }
//     });
// });