{/* <script type="text/javascript"> */}


  function autoFill(songCategories) {
    const prepopulatedValues = {
      popVsRock: ["pop song, Pop","rock song, Rock"],
      rockVsHiphop: ["rock song, Rock","hip hop song, Hiphop"],
      mixedGenres: ["disco song, Disco","indie song, Indie"]
    }
    document.getElementsByName('query_a')[0].value = prepopulatedValues[songCategories][0];
    document.getElementsByName('query_b')[0].value = prepopulatedValues[songCategories][1];
 
    // var radioElements = document.getElementsByName("input3");

    // for (var i=0; i<radioElements.length; i++) {
    //   if (radioElements[i].getAttribute('value') == 'Radio3') {
    //     radioElements[i].checked = true;
    //   }
    // }
  }
// </script>
  