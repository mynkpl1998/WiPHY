fetch('http:/localhost:5000/query/devices').then((response) => {
	return response.json()
}).then((response) => {
	console.log(response)
})
