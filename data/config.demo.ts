export const config = {
    // laf云后台的appid
    LAF_APPID: "your_laf_appid",
    // 云函数getTempCOS调用得到
    // 如果报错403，需要加laf环境变量，DEV_IPS="xxx.xxx.xxx.xxx"，其中xxx为调用本段代码的机器公网ip
    COS_KEY: {
        "ExpiredTime": 1687775333,
        "Expiration": "2023-06-26T10:28:53Z",
        "Credentials": {
            "Token": "your_token",
            "TmpSecretId": "your_secret_id",
            "TmpSecretKey": "your_secret_key"
        },
        "RequestId": "0c56a822-7d46-4256-8446-8c038ac32ce6"
    }
}
