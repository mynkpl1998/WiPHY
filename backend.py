from flask import Flask, request, jsonify, after_this_request
from rtlsdr import RtlSdr

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/query/devices', methods=['GET', 'POST'])
def get_available_devices():

	@after_this_request
	def add_header(response):
		response.headers.add('Access-Control-Allow-Origin', '*')
		return response

	# GET Request
	if request.method == 'GET':
		devices_serial = RtlSdr.get_device_serial_addresses()
		message = []
		for dev_id, dev_serial in enumerate(devices_serial):
			dev_dict = {}
			dev_dict[dev_id] = devices_serial
			message.append(dev_dict)
		return jsonify(message)

if __name__ ==  "__main__":
	
	app.run()
