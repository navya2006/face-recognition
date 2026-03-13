function registerUser(){

let name = document.getElementById("name").value
let id = document.getElementById("id").value

fetch("/register",{

method:"POST",

headers:{
"Content-Type":"application/json"
},

body:JSON.stringify({
name:name,
id:id
})

})

.then(response => response.json())

.then(data => {

alert(data.message)

})

}


function recognizeFace(){

fetch("/recognize")

.then(response => response.json())

.then(data => {

document.getElementById("result").innerHTML =
"Detected Person: " + data.person

})

}