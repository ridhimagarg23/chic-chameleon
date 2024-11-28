const analyzeButton = document.getElementById("analyzeButton");
const popup = document.getElementById("popup");
const extraInputs = document.getElementById("extraInputs");
const maleInputs = document.getElementById("maleInputs");
const continueAnalyze = document.getElementById("continueAnalyze");
const results = document.getElementById("results");
const uploadedPhoto = document.getElementById("uploadedPhoto");
const genderField = document.getElementById("gender");
const skinToneField = document.getElementById("skinTone");
const colorBox = document.getElementById("colorBox");
const bodyShapeField = document.getElementById("bodyShape");

analyzeButton.addEventListener("click", () => {
    const fileInput = document.getElementById("photoUpload");
    if (fileInput.files.length === 0) {
        alert("Please upload a photo first.");
        return;
    }
    
      const formData = new FormData();
      formData.append("photo", fileInput.files[0]);

      fetch("/analyze", {
      method: "POST",
      body: formData
      })
    .then(response => {
        if (!response.ok) {
            throw new Error("Network response was not ok");
        }
        return response.json();
    })
    .then(data => {
        if (data.gender === "female") {
            extraInputs.classList.remove("hidden");
            maleInputs.classList.add("hidden");
        } else if (data.gender === "male") {
            maleInputs.classList.remove("hidden");
            extraInputs.classList.add("hidden");
        }
        popup.classList.remove("hidden");
    })
    .catch(error => {
        console.error("Error:", error);
        alert("An error occurred during analysis. Please try again.");
    });
});

continueAnalyze.addEventListener("click", () => {
      popup.classList.add("hidden");
      results.classList.remove("hidden");
      fetch("/final_analyze")
          .then(response => {
              if (!response.ok) {
                  throw new Error("Network response was not ok");
              }
              return response.json();
          })
          .then(data => {
              uploadedPhoto.src = data.photo_url;
              genderField.textContent = data.gender;
              skinToneField.textContent = data.skin_tone;
              colorBox.style.backgroundColor = data.skin_color;
              bodyShapeField.textContent = data.body_shape;
          })
          .catch(error => {
              console.error("Error:", error);
              alert("An error occurred during final analysis. Please try again.");
          });
  });