import base64

from androguard.core.bytecodes.apk import APK
from io import BytesIO

from PIL import Image

from appclusters.preprocessing.permissionparser import PermissionParser


def apk_to_report_saver(apk_file, report_saver):
    apk = APK(apk_file)

    try:
        app_icon_file = apk.get_app_icon()
        app_icon_data = apk.get_file(app_icon_file)

        size = (256, 256)

        buffered = BytesIO()
        im = Image.open(BytesIO(app_icon_data))
        im = im.resize(size, Image.ANTIALIAS)
        im.save(buffered, "PNG")

        app_icon_b64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')
    except:
        app_icon_b64 = ""

    package_name = apk.get_package()
    app_name = apk.get_app_name()

    report_saver.package_name = package_name
    report_saver.app_name = app_name
    report_saver.version = apk.get_androidversion_code()
    report_saver.app_icon = app_icon_b64

    permission_parser = PermissionParser(mode='groups')
    permission_values = permission_parser.transform(apk.get_permissions()).flatten().tolist()
    permission_labels = permission_parser.labels()
    report_saver.permissions_actual = {permission_labels[i]: bool(v) for i, v in enumerate(permission_values)}

    return package_name, app_name
