# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 12:44:52 2022

@author: nurbuketeker
"""

import tensorflow as tf
import torch
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.preprocessing import LabelEncoder


device_name = tf.test.gpu_device_name()

device = torch.device("cpu")
print('GPU:', torch.cuda.get_device_name(0))

from transformers import AutoModelForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

labelencoder = LabelEncoder()

class_dictionary={0 :"ATIS",2:"ACID", 1: "Banking",3:"CLINC"}
 
def getModel():
    filename =  "big_model"
    pytorch_model = AutoModelForSequenceClassification.from_pretrained(filename)
    return pytorch_model


def getModelPrediction(text,pytorch_model):
    test_texts_ = [text]
    
    input_ids = []
    attention_masks = []
    
    for text in test_texts_:
        encoded_dict = tokenizer.encode_plus(
                            text,                     
                            add_special_tokens = True, 
                            max_length = 20,          
                            pad_to_max_length = True,
                            return_attention_mask = True,  
                            return_tensors = 'pt',   
                       )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
            
   
        
    test_labels_ = labelencoder.fit_transform( [1])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(test_labels_.tolist())
    
    batch_size = 32  
    
    prediction_data = TensorDataset(input_ids, attention_masks,labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    
    print('Prediction started on test data')
    pytorch_model.eval()
    predictions , true_labels = [], []
    
    
    for batch in prediction_dataloader:
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_input_mask, b_labels = batch
    
      with torch.no_grad():
          outputs = pytorch_model(b_input_ids, token_type_ids=None, 
                          attention_mask=b_input_mask)
    
      logits = outputs[0]
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to("cpu").numpy()
      
      predictions.append(logits)
      true_labels.append(label_ids)
    
    print('Prediction completed')
    
    prediction_set = []
    
    for i in range(len(true_labels)):
      pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
      prediction_set.append(pred_labels_i)
    
    prediction_scores = [item for sublist in prediction_set for item in sublist]
    return prediction_scores

def getIntent( text):
    pytorch_model = getModel()
    prediction_scores = getModelPrediction(text,pytorch_model)
    print(prediction_scores)
    #prediction_class = class_dictionary[prediction_scores[0]]
    return all_dict[prediction_scores[0]]



all_dict ={252: 'atis_flight',
 250: 'atis_airfare',
 251: 'atis_airline',
 253: 'atis_ground_service',
 248: 'atis_abbreviation',
 249: 'atis_aircraft',
 262: 'card_arrival',
 264: 'card_linking',
 282: 'exchange_rate',
 268: 'card_payment_wrong_exchange_rate',
 284: 'extra_charge_on_statement',
 296: 'pending_cash_withdrawal',
 286: 'fiat_currency_support',
 263: 'card_delivery_estimate',
 255: 'automatic_top_up',
 265: 'card_not_working',
 283: 'exchange_via_app',
 291: 'lost_or_stolen_card',
 246: 'age_limit',
 299: 'pin_blocked',
 305: 'top_up_by_bank_transfer_charge',
 297: 'pending_top_up',
 259: 'cancel_transfer',
 309: 'top_up_limits',
 323: 'wrong_amount_of_cash_received',
 266: 'card_payment_fee_charged',
 315: 'transfer_not_received_by_recipient',
 303: 'supported_cards_and_currencies',
 290: 'getting_virtual_card',
 261: 'card_acceptance',
 310: 'top_up_reverted',
 257: 'balance_not_updated_after_cheque_or_cash_deposit',
 267: 'card_payment_not_recognised',
 280: 'edit_personal_details',
 322: 'why_verify_identity',
 317: 'unable_to_verify_identity',
 288: 'get_physical_card',
 321: 'visa_or_mastercard',
 311: 'topping_up_by_card',
 279: 'disposable_card_limits',
 273: 'compromised_card',
 254: 'atm_support',
 278: 'direct_debit_payment_not_recognised',
 294: 'passcode_forgotten',
 276: 'declined_cash_withdrawal',
 295: 'pending_card_payment',
 292: 'lost_or_stolen_phone',
 301: 'request_refund',
 277: 'declined_transfer',
 244: 'Refund_not_showing_up',
 275: 'declined_card_payment',
 298: 'pending_transfer',
 304: 'terminate_account',
 269: 'card_swallowed',
 312: 'transaction_charged_twice',
 319: 'verify_source_of_funds',
 316: 'transfer_timing',
 302: 'reverted_card_payment?',
 272: 'change_pin',
 258: 'beneficiary_not_allowed',
 313: 'transfer_fee_charged',
 300: 'receiving_money',
 285: 'failed_transfer',
 314: 'transfer_into_account',
 320: 'verify_top_up',
 289: 'getting_spare_card',
 307: 'top_up_by_cash_or_cheque',
 293: 'order_physical_card',
 324: 'wrong_exchange_rate_for_cash_withdrawal',
 287: 'get_disposable_virtual_card',
 308: 'top_up_failed',
 256: 'balance_not_updated_after_bank_transfer',
 271: 'cash_withdrawal_not_recognised',
 281: 'exchange_charge',
 306: 'top_up_by_card_charge',
 245: 'activate_my_card',
 270: 'cash_withdrawal_charge',
 260: 'card_about_to_expire',
 247: 'apple_pay_or_google_pay',
 318: 'verify_my_identity',
 274: 'country_support',
 11: '108',
 110: '62',
 46: '14',
 70: '26',
 7: '104',
 2: '10',
 104: '57',
 97: '50',
 67: '23',
 53: '146',
 48: '141',
 58: '150',
 91: '45',
 49: '142',
 71: '27',
 79: '34',
 136: '86',
 65: '21',
 142: '91',
 113: '65',
 68: '24',
 22: '118',
 126: '77',
 93: '47',
 103: '56',
 132: '82',
 145: '94',
 17: '113',
 98: '51',
 94: '48',
 82: '37',
 69: '25',
 129: '8',
 115: '67',
 61: '18',
 131: '81',
 112: '64',
 72: '28',
 56: '149',
 143: '92',
 35: '13',
 9: '106',
 21: '117',
 36: '130',
 108: '60',
 123: '74',
 106: '59',
 64: '20',
 89: '43',
 28: '123',
 105: '58',
 19: '115',
 25: '120',
 99: '52',
 128: '79',
 6: '103',
 47: '140',
 81: '36',
 40: '134',
 8: '105',
 150: '99',
 121: '72',
 45: '139',
 3: '100',
 4: '101',
 135: '85',
 90: '44',
 41: '135',
 77: '32',
 39: '133',
 111: '63',
 102: '55',
 38: '132',
 140: '9',
 83: '38',
 15: '111',
 96: '5',
 42: '136',
 54: '147',
 0: '0',
 139: '89',
 63: '2',
 57: '15',
 74: '3',
 100: '53',
 86: '40',
 31: '126',
 109: '61',
 13: '11',
 5: '102',
 10: '107',
 141: '90',
 124: '75',
 14: '110',
 149: '98',
 29: '124',
 148: '97',
 92: '46',
 34: '129',
 1: '1',
 95: '49',
 16: '112',
 127: '78',
 20: '116',
 33: '128',
 137: '87',
 78: '33',
 52: '145',
 50: '143',
 80: '35',
 76: '31',
 27: '122',
 84: '39',
 30: '125',
 51: '144',
 44: '138',
 134: '84',
 101: '54',
 138: '88',
 62: '19',
 114: '66',
 32: '127',
 120: '71',
 119: '70',
 60: '17',
 75: '30',
 147: '96',
 118: '7',
 24: '12',
 43: '137',
 144: '93',
 125: '76',
 23: '119',
 107: '6',
 73: '29',
 116: '68',
 37: '131',
 146: '95',
 55: '148',
 122: '73',
 66: '22',
 130: '80',
 87: '41',
 26: '121',
 133: '83',
 117: '69',
 18: '114',
 85: '4',
 59: '16',
 12: '109',
 88: '42',
 151: 'INFO_ADD_REMOVE_INSURED',
 152: 'INFO_ADD_REMOVE_VEHICLE',
 153: 'INFO_ADD_VEHICLE_PROPERTY_PAPERLESS_BILLING',
 154: 'INFO_AGT_NOT_RESPONDING',
 155: 'INFO_AMT_DUE',
 156: 'INFO_AUTO_COV_QUESTION',
 157: 'INFO_AUTO_INS_CANADA',
 158: 'INFO_AUTO_POLICY_CANT_SEE_IN_ACCT',
 159: 'INFO_AUTO_PYMT_CANCEL',
 160: 'INFO_AUTO_PYMT_SCHEDULE',
 163: 'INFO_BILL_DUE_DATE',
 161: 'INFO_BILLING_ACCT_NUM',
 162: 'INFO_BILLING_DEPT_CONTACT',
 164: 'INFO_BUSINESS_POLICY_CANT_SEE',
 165: 'INFO_CANCEL_FEE',
 166: 'INFO_CANCEL_INS_POLICY',
 167: 'INFO_CANT_SEE_FARM_RANCH_POLICY',
 168: 'INFO_CANT_SEE_POLICY',
 169: 'INFO_CAREERS',
 170: 'INFO_CHANGE_BANK_ACCT',
 171: 'INFO_CHANGE_USERID',
 172: 'INFO_CL_ADJUSTER_INFO',
 173: 'INFO_CL_CLAIM_FILED',
 174: 'INFO_CL_COMPLAINT',
 175: 'INFO_CL_DOCS_EMAIL',
 176: 'INFO_CL_DOCS_FAX',
 177: 'INFO_CL_FNOL',
 178: 'INFO_CL_STATUS',
 179: 'INFO_CL_UPDATE_INFO',
 180: 'INFO_COMBINE_PYMTS',
 181: 'INFO_CREDIT_CARD_CHANGE_NUM',
 182: 'INFO_CREDIT_CARD_FEE',
 183: 'INFO_DEC_PAGE_NEEDED',
 184: 'INFO_DELETE_DUPE_PYMT',
 185: 'INFO_DIFFERENT_AMTS',
 186: 'INFO_DISCOUNTS',
 187: 'INFO_DO_NOT_CONTACT',
 188: 'INFO_ERS',
 189: 'INFO_FIND_AGENT',
 190: 'INFO_FORGOT_EMAIL',
 191: 'INFO_FORGOT_PASSWORD',
 192: 'INFO_FORGOT_USERID',
 193: 'INFO_GEN_POLICY_COV_QUESTION',
 194: 'INFO_GET_A_QUOTE_CFR',
 195: 'INFO_GET_A_QUOTE_OTHER',
 196: 'INFO_GLASS_COV',
 197: 'INFO_HANDLING_FEE_REMOVE',
 198: 'INFO_HEALTH_INS_QUOTE',
 199: 'INFO_INS_CARD_PROOF',
 200: 'INFO_INS_NOT_AVAILABLE',
 201: 'INFO_LETTER_OF_EXPERIENCE',
 202: 'INFO_LIFE_BENEFICIARY_CHANGE',
 203: 'INFO_LIFE_CASH_OUT',
 204: 'INFO_LIFE_INCR_COV',
 205: 'INFO_LIFE_POLICY_AMT_DUE',
 206: 'INFO_LIFE_POLICY_AUTO_PYMT',
 207: 'INFO_LIFE_POLICY_CANCEL',
 208: 'INFO_LIFE_POLICY_CANNOT_SEE',
 209: 'INFO_LIFE_REFUND',
 210: 'INFO_LIFE_UPDATE_CONTACT_INFO',
 212: 'INFO_LOG_OUT',
 211: 'INFO_LOGIN_ERROR',
 213: 'INFO_MAIL_PYMT_ADDRESS',
 214: 'INFO_MAKE_PYMT',
 215: 'INFO_MEXICO_AUTO_INS',
 216: 'INFO_NAME_CHANGE',
 217: 'INFO_ONE_TIME_PYMT',
 218: 'INFO_OPERATING_AREA',
 219: 'INFO_PAPERLESS_DOCS_SETUP',
 220: 'INFO_PAPERLESS_DOCS_STOP',
 221: 'INFO_PAY_LIFE_INS',
 222: 'INFO_PHONE_NUM_INTERNATIONAL',
 223: 'INFO_POLICY_DOC_NEEDED',
 224: 'INFO_POLICY_NUM',
 225: 'INFO_POLICY_TRANS_TO_RENTAL',
 226: 'INFO_PREPAID_CARD_PYMT',
 227: 'INFO_PYMT_CONFIRM',
 228: 'INFO_PYMT_DUEDATE_CHANGE',
 229: 'INFO_PYMT_HISTORY',
 230: 'INFO_PYMT_NOT_ONTIME',
 231: 'INFO_PYMT_PROCESS_CHANGE',
 232: 'INFO_PYMT_SETUP_AUTO_PYMT',
 233: 'INFO_REFUND_CHECK',
 234: 'INFO_REINSTATE_INS_POLICY',
 235: 'INFO_RIDESHARE_COV',
 236: 'INFO_SET_UP_ACCT',
 237: 'INFO_SPEAK_TO_REP',
 238: 'INFO_TRANSFER_ACCT_BALANCE',
 239: 'INFO_UPDATE_CONTACT_INFO',
 240: 'INFO_UPDATE_EMAIL',
 241: 'INFO_UPDATE_LIENHOLDER',
 242: 'INFO_UPDATE_PHONE_NUM',
 243: 'INFO_WHY_WAS_POLICY_CANCELLED'}